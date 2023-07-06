"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import paddle
import paddle.nn.functional as F

# import lavis.common.dist_utils as dist_utils
from lavis_paddle.common.utils import is_url
from lavis_paddle.common.logger import MetricLogger
from lavis_paddle.models.base_model import BaseModel
from paddlenlp.transformers.bert.configuration import BertConfig
from lavis_paddle.models.blip2_models.Qformer import BertLMHeadModel
from lavis_paddle.models.eva_vit import create_eva_vit_g
from paddlenlp.transformers import AutoTokenizer


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self):
        return paddle.amp.auto_cast(dtype='float16')

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        # todo check dropout
        # encoder_config.attention_probs_dropout_prob = 0
        # encoder_config.hidden_dropout_prob = 0
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = paddle.create_parameter(
            shape=(1, num_query_token, encoder_config.hidden_size),
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=encoder_config.initializer_range)
        )
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):

        visual_encoder = create_eva_vit_g(img_size, drop_path_rate, use_grad_checkpoint, precision)

        ln_vision = paddle.nn.LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            raise RuntimeError("checkpoint url is not support")
        elif os.path.isfile(url_or_filename):
            checkpoint = paddle.load(url_or_filename)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.set_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pd",
        )
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = paddle.concat(text_embeds, axis=0)
    text_ids = paddle.concat(text_ids, axis=0)
    text_atts = paddle.concat(text_atts, axis=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, axis=-1)

        vit_feats.append(vit_feat)
        image_embeds.append(image_embed)

    vit_feats = paddle.concat(vit_feats, axis=0)
    image_embeds = paddle.concat(image_embeds, axis=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = paddle.stack(sims_matrix, axis=0)

    score_matrix_i2t = paddle.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    )

    num_tasks = 1
    # num_tasks = dist_utils.get_world_size()
    rank = 0
    # rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, axis=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        )
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = paddle.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    )

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, axis=0)
        image_inputs = vit_feats[topk_idx]
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.numpy(), score_matrix_t2i.numpy()


