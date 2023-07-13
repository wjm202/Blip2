
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from altair import value
import paddle
import paddle.nn as nn
import paddle.distributed as dist

from lavis_paddle.common.registry import registry
from lavis_paddle.models.base_model import all_gather_with_grad, concat_all_gather
from lavis_paddle.models.blip2_models.blip2 import Blip2Base, compute_sim_matrix, disabled_train
#from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

@registry.register_model('blip2')
@registry.register_model('blip2_feature_extractor')
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {'pretrain':
        'configs/models/blip2/blip2_pretrain.yaml', 'pretrain_vitL':
        'configs/models/blip2/blip2_pretrain_vitL.yaml', 'coco':
        'configs/models/blip2/blip2_coco.yaml'}

    def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate
        =0, use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=
        True, num_query_token=32, cross_attention_freq=2, embed_dim=256,
        max_txt_len=32):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        logging.info('freeze vision encoder' + vit_model)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint,
            vit_precision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.stop_gradient = True
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info('freeze vision encoder')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token,
            self.visual_encoder.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        #import pdb;pdb.set_trace()
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if '_query' in name:
                key_orig = name.replace('_query', '')
                param.copy_(state_dict[key_orig], False) ### problem
        self.vision_proj = paddle.incubate.nn.FusedLinear(in_features=self.Qformer.config
            .hidden_size, out_features=embed_dim)
        self.text_proj = paddle.incubate.nn.FusedLinear(in_features=self.Qformer.config.
            hidden_size, out_features=embed_dim)
        self.itm_head = paddle.incubate.nn.FusedLinear(in_features=self.Qformer.config.
            hidden_size, out_features=2)
        self.temp = self.create_parameter(
            shape=(1, ), default_initializer=paddle.nn.initializer.Constant(value=0.07))
        self.max_txt_len = max_txt_len

    def forward(self, samples):
        samples = samples[0]
        text = samples['text_input']
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        # text=[]
        # with open("/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/text_input.txt") as f: #debug
        #     while 1:
        #         line = f.readline()
        #         if not line:
        #             break
        #         if line[-1]=="\n":
        #             line=line[:-1]
        #         text.append(line)
        # import numpy as np
        # image = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/image.npy"))
        # image_embeds = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/image_embeds.npy"))
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype="int64")        
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])
        
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, use_cache=True, return_dict=True)
        image_feats = paddle.nn.functional.normalize(x=self.vision_proj(
            query_output.last_hidden_state), axis=-1)
        
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        text_tokens = self.tokenizer(text, 
                                     padding='max_length', 
                                     truncation=True, 
                                     max_length=self.max_txt_len, 
                                     return_attention_mask=True, 
                                     return_tensors="pd"
                        )
        text_output = self.Qformer.bert(text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask, return_dict=True)
        text_feat = paddle.nn.functional.normalize(self.text_proj(
            text_output.last_hidden_state[:, 0, :]), axis=-1)
        
        ###============== Image-text Contrastive ===================###
        # image_feats_all = image_feats
        # text_feat_all = text_feat
        image_feats_all = concat_all_gather(image_feats)
        text_feat_all = concat_all_gather(text_feat)
        sim_q2t = paddle.matmul(image_feats.unsqueeze(axis=1), 
            text_feat_all.unsqueeze(axis=-1)).squeeze()
        sim_i2t = sim_q2t.max(axis=-1)
        sim_i2t = sim_i2t / self.temp
        sim_t2q = paddle.matmul(x=text_feat.unsqueeze(axis=1).unsqueeze(
            axis=1), y=image_feats_all.transpose(perm=[0, 2, 1])).squeeze()
        sim_t2i = sim_t2q.max(axis=-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = image.shape[0]
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        targets = paddle.linspace(start=rank * bs, stop=rank * bs + bs - 1,
            num=bs).astype(int)
        #import pdb;pdb.set_trace()
        one_hot_label = paddle.nn.functional.one_hot(targets, num_classes=sim_i2t.shape[1])
        smooth_label = paddle.nn.functional.label_smooth(label=one_hot_label, epsilon=0.1)
        loss_itc = (paddle.nn.functional.cross_entropy(
            input=sim_i2t, label=smooth_label, soft_label=True) + 
                    paddle.nn.functional.cross_entropy(
            input=sim_t2i, label=smooth_label, soft_label=True)) / 2
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)   
        # import pdb
        # pdb.set_trace()  
        # text_input_ids_world = text_tokens.input_ids
        # text_attention_mask_world = text_tokens.attention_mask
        # image_embeds_world = image_embeds
        with paddle.no_grad():
            weights_t2i = paddle.nn.functional.softmax(x=sim_t2i, axis=1
                ) + 0.0001
            weights_t2i_list= paddle.chunk(weights_t2i,chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_t2i_list[rank].fill_diagonal_(value=0)
            weights_t2i = paddle.concat(weights_t2i_list,axis=-1)
            weights_i2t = paddle.nn.functional.softmax(x=sim_i2t, axis=1
                ) + 0.0001
            weights_i2t_list= paddle.chunk(weights_i2t,chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_i2t_list[rank].fill_diagonal_(value=0)
            weights_i2t = paddle.concat(weights_i2t_list,axis=-1)
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_t2i[b], num_samples=1).item(
                )
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = paddle.stack(x=image_embeds_neg, axis=0)
        # image_embeds_neg = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/image_embeds_neg.npy"))
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_i2t[b], num_samples=1).item(
                )
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = paddle.stack(x=text_ids_neg, axis=0)
        text_atts_neg = paddle.stack(x=text_atts_neg, axis=0)
        # text_ids_neg = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/text_ids_neg.npy"))
        # text_atts_neg = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/text_atts_neg.npy"))
        text_ids_all = paddle.concat(x=[text_tokens.input_ids, text_tokens.
            input_ids, text_ids_neg], axis=0)
        text_atts_all = paddle.concat(x=[text_tokens.attention_mask,
            text_tokens.attention_mask, text_atts_neg], axis=0)
        query_tokens_itm = self.query_tokens.expand(shape=[text_ids_all.
            shape[0], -1, -1])
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        query_atts_itm = paddle.ones(shape=query_tokens_itm.shape[:-1],
            dtype='int64')
        attention_mask_all = paddle.concat(x=[query_atts_itm, text_atts_all
            ], axis=1)
        image_embeds_all = paddle.concat(x=[image_embeds, image_embeds_neg,
            image_embeds], axis=0)
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        image_atts_all = paddle.ones(shape=image_embeds_all.shape[:-1],
            dtype='int64')
        output_itm = self.Qformer.bert(text_ids_all, query_embeds=
            query_tokens_itm, attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all, encoder_attention_mask=
            image_atts_all, return_dict=True)
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.
            shape[1], :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(axis=1)
        
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        itm_labels = paddle.concat([paddle.ones([bs], dtype='int64'),
            paddle.zeros([2 * bs], dtype='int64')], axis=0)
        loss_itm = paddle.nn.functional.cross_entropy(input=logits, label=
            itm_labels)
        ##================= Image Captioning ========================##
        
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, (0)] = self.tokenizer.bos_token_id
        labels = masked_fill(decoder_input_ids, decoder_input_ids == self.tokenizer.pad_token_id, -100)
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype='int64')
        attention_mask = paddle.concat(x=[query_atts, text_tokens.
            attention_mask], axis=1)
        #import pdb;pdb.set_trace()
        lm_output = self.Qformer(decoder_input_ids, attention_mask=
            attention_mask, past_key_values=query_output.past_key_values,
            return_dict=True, labels=labels)
        loss_lm = lm_output.loss
        return dict(loss=loss_itc + loss_itm + loss_lm, loss_itc=
            loss_itc, loss_itm=loss_itm, loss_lm=loss_lm)

    @paddle.no_grad()
    def generate(self, samples, use_nucleus_sampling=False, num_beams=3,
        max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples['image']
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if not use_nucleus_sampling:
            """Class Method: *.repeat_interleave, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            image_embeds = image_embeds.repeat_interleave(num_beams, axis=0)
        else:
            num_beams = 1
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype='int64')
        model_kwargs = {'encoder_hidden_states': image_embeds,
            'encoder_attention_mask': image_atts}
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        input_ids = paddle.empty(shape=[image.shape[0], 1], dtype='int64').fill_(value=self.tokenizer.bos_token_id)
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0
            ], -1, -1])
        outputs = self.Qformer.generate(input_ids=input_ids, query_embeds=
            query_tokens, max_length=max_length, min_length=min_length,
            num_beams=num_beams, do_sample=use_nucleus_sampling, top_p=
            top_p, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=
            self.tokenizer.pad_token_id, **model_kwargs)
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens
            =True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype='int64')
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0
            ], -1, -1])
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask, return_dict=True)
        return text_output.last_hidden_state[:, (0), :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        image_atts = paddle.ones(shape=image_inputs.shape[:-1], dtype='int64'
            )
        query_tokens = self.query_tokens.expand(shape=[image_inputs.shape[0
            ], -1, -1])
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype='int64'
            )
        attention_mask = paddle.concat(x=[query_atts, text_atts], axis=1)
        output_itm = self.Qformer.bert(text_ids, query_embeds=query_tokens,
            attention_mask=attention_mask, encoder_hidden_states=
            image_inputs, encoder_attention_mask=image_atts, return_dict=True)
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens.shape
            [1], :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, (1)].mean(axis=1)
        return itm_logit

    @paddle.no_grad()
    def extract_features(self, samples, mode='multimodal'):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get('image')
        caption = samples.get('text_input')
        assert mode in ['image', 'text', 'multimodal'
            ], "mode must be one of 'image', 'text', 'multimodal'"
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None
        if mode == 'image':
            assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image)
                    )
            image_embeds_frozen = image_embeds_frozen.astype(dtype='float32')
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            image_atts = paddle.ones(shape=image_embeds_frozen.shape[:-1],
                dtype='int64')
            query_tokens = self.query_tokens.expand(shape=[
                image_embeds_frozen.shape[0], -1, -1])
            query_output = self.Qformer.bert(query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts, return_dict=True)
            image_embeds = query_output.last_hidden_state
            image_features = paddle.nn.functional.normalize(x=self.
                vision_proj(image_embeds), axis=-1)
        elif mode == 'text':
            assert caption is not None, "text input is None for mode 'text' or 'multimodal'"
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            text = self.tokenizer(caption, return_tensors='pt', padding=True
                )
            text_output = self.Qformer.bert(text.input_ids, attention_mask=
                text.attention_mask, return_dict=True)
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = paddle.nn.functional.normalize(x=text_features,
                axis=-1)
        elif mode == 'multimodal':
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image)
                    )
            image_embeds_frozen = image_embeds_frozen.astype(dtype='float32')
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            image_atts = paddle.ones(shape=image_embeds_frozen.shape[:-1],
                dtype='int64')
            query_tokens = self.query_tokens.expand(shape=[
                image_embeds_frozen.shape[0], -1, -1])
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype=
                'int64')
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            text = self.tokenizer(caption, return_tensors='pt', padding=True
                )
            attention_mask = paddle.concat(x=[query_atts, text.
                attention_mask], axis=1)
            output = self.Qformer.bert(text.input_ids, query_embeds=
                query_tokens, attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts, return_dict=True)
            multimodal_embeds = output.last_hidden_state[:, :query_tokens.
                shape[1], :]
        return dict(image_embeds=image_embeds,
            image_embeds_proj=image_features, text_embeds=text_embeds,
            text_embeds_proj=text_features, multimodal_embeds=multimodal_embeds
            )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get('vit_model', 'eva_clip_g')
        img_size = cfg.get('image_size')
        num_query_token = cfg.get('num_query_token')
        cross_attention_freq = cfg.get('cross_attention_freq', 2)
        drop_path_rate = cfg.get('drop_path_rate', 0)
        use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)
        vit_precision = cfg.get('vit_precision', 'fp16')
        freeze_vit = cfg.get('freeze_vit', True)
        max_txt_len = cfg.get('max_txt_len', 32)
        model = cls(vit_model=vit_model, img_size=img_size, drop_path_rate=
            drop_path_rate, use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision, freeze_vit=freeze_vit,
            num_query_token=num_query_token, cross_attention_freq=
            cross_attention_freq, max_txt_len=max_txt_len)

        model.load_checkpoint_from_config(cfg)
        
        import numpy as np
        model.state_dict()['query_tokens'].set_value(paddle.to_tensor(np.load("/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/query_token_stage1.npy")))
        text_proj = paddle.load("/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/text_proj.pdparams")
        vision_proj = paddle.load("/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/vision_proj.pdparams")
        itm_head = paddle.load("/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/itm_head.pdparams")
        model.state_dict()['text_proj.weight'].set_value(text_proj['text_proj.weight'])
        model.state_dict()['text_proj.bias'].set_value(text_proj['text_proj.bais'])
        model.state_dict()['vision_proj.weight'].set_value(vision_proj['vision_proj.weight'])
        model.state_dict()['vision_proj.bias'].set_value(vision_proj['vision_proj.bais'])
        model.state_dict()['itm_head.weight'].set_value(itm_head['itm_head.weight'])
        model.state_dict()['itm_head.bias'].set_value(itm_head['itm_head.bais'])
        model.state_dict()['Qformer.bert.embeddings.word_embeddings.weight'].set_value(paddle.to_tensor(np.load("word_embeddings.npy")))
        # model.state_dict()['Qformer.cls.predictions.decoder.weight'].set_value(paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/Qformer.cls.predictions.decoder.weight.npy")))
        return model


    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test
        return compute_sim_matrix(model=self, data_loader=data_loader,
            k_test=k_test)
