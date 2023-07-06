"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import paddle
import paddle.nn as nn

from lavis_paddle.common.registry import registry
from lavis_paddle.models.blip2_models.blip2 import Blip2Base, disabled_train
from paddlenlp.transformers import AutoTokenizer
from lavis_paddle.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig


@registry.register_model("blip2_opt")
class Blip2OPTPaddle(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
    ):  
        vit_precision="fp32"
        super().__init__()
        # self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.stop_gradient = True
        #     self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 1408
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        # self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
        # for name, param in self.opt_model.named_parameters():
        #     param.stop_gradient = True
        # self.eos_token_id = self.opt_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]
        # self.opt_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        # )

        # self.max_txt_len = max_txt_len
        # self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pd", return_attention_mask=True)
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
    
    def forward(self, samples):
        # samples = self.prepare_samples(samples)
        text=[]
        with open("/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/text_input.txt") as f: #debug
            while 1:
                line = f.readline()
                if not line:
                    break
                if line[-1]=="\n":
                    line=line[:-1]
                text.append(line)
        text=[t + "\n" for t in text]
        import numpy as np
        image = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/image.npy"))[:64]
        image_embeds = paddle.to_tensor(np.load("/paddle/workspace/wjm/LAVIS/image_embeds.npy"))
        # image_embeds=image_embeds[:64]
        text=text[:64]
        # samples = samples[0]
        # image = samples["image"]
         
        # with paddle.amp.auto_cast(level='O2'):
        # image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.astype("float32")
        # text=[t + "\n" for t in samples["text_input"]]
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype='int64')
        query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

#         inputs_opt = self.opt_proj(query_output.last_hidden_state)
#         atts_opt = paddle.ones(inputs_opt.shape[:-1], dtype='int64')

#         self.opt_tokenizer.padding_side = "right"


#         # with paddle.amp.auto_cast(level='O2'):
#         opt_tokens = self.opt_tokenizer(
#                     text,
#                     return_tensors="pd",
#                     padding="longest",
#                     truncation=True,
#                     max_length=self.max_txt_len,
#                     return_attention_mask=True
#                 )
# #        targets = opt_tokens.input_ids.masked_fill(
# #            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
# #        )

#         targets = opt_tokens.input_ids*(1-(opt_tokens.input_ids == self.opt_tokenizer.pad_token_id).astype(opt_tokens.input_ids.dtype)) + (opt_tokens.input_ids == self.opt_tokenizer.pad_token_id).astype(opt_tokens.input_ids.dtype)*(-100)
#         if self.prompt:
#             targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

#         empty_targets = (
#             paddle.ones(atts_opt.shape, dtype='int64').fill_(-100)
#         )
#         targets = paddle.concat([empty_targets, targets], axis=1)

#         inputs_embeds = self.opt_model.embeddings.word_embeddings(opt_tokens.input_ids)
#         inputs_embeds = paddle.concat([inputs_opt, inputs_embeds], axis=1)
#         attention_mask = paddle.concat([atts_opt, opt_tokens.attention_mask], axis=1)

#         # with paddle.amp.auto_cast(level='O2'):
#         #paddle.save({'inputs_embeds':inputs_embeds, 'attention_mask':attention_mask, 'targets':targets}, 'opt_it.pdparams')
#         outputs = self.opt_model(
#                     inputs_embeds=inputs_embeds,
#                     attention_mask=attention_mask,
#                     return_dict=True,
#                     labels=targets,
#                 )
        loss = query_output.last_hidden_state.sum()
        return {"loss": loss}

    @paddle.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
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
        image = samples[1]["image"] #[16, 3, 364, 364]

        with paddle.amp.auto_cast():
            image_embeds = self.ln_vision(self.visual_encoder(image)) #16 677
            image_atts = paddle.ones(image_embeds.shape[:-1], dtype='int64') #16 677

            query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1]) #[16, 32, 768]
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = paddle.ones(inputs_opt.shape[:-1], dtype='int64')

            prompt = self.prompt
            prompt = [prompt] * image.shape[0]

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pd", return_attention_mask=True)
            input_ids = opt_tokens.input_ids
            attention_mask = paddle.concat([atts_opt, opt_tokens.attention_mask], axis=1) # 16,32 16, 4->16,32

            if use_nucleus_sampling: # eval:false 
                query_embeds = inputs_opt.repeat_interleave(num_captions, axis=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, axis=0)
            
            outputs = self.opt_model.generate(
                input_ids=input_ids, #16,4
                query_embeds=query_embeds, #80 32 2560
                attention_mask=attention_mask, #16 36
                do_sample=use_nucleus_sampling,
                decode_strategy="beam_search", # align to torch
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,#30
                min_length=min_length-4, # 8
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[0], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)
        # import numpy as np
        # model.state_dict()['opt_proj.weight'].set_value(np.load("/paddle/workspace/wjm/LAVIS/opt_proj_weight.npy"))
        # model.state_dict()['opt_proj.bias'].set_value(np.load("/paddle/workspace/wjm/LAVIS/opt_proj_bias.npy"))
        
        

        return model
