"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
# import torch
import paddle

# import torch.backends.cudnn as cudnn

import lavis_paddle.tasks as tasks
from lavis_paddle.common.config import Config
# from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis_paddle.common.logger import setup_logger

from lavis_paddle.common.registry import registry
from lavis_paddle.common.utils import now

# imports modules for registration
from lavis_paddle.datasets.builders import *
from lavis_paddle.models import *
from lavis_paddle.processors import *
from lavis_paddle.runners import *
from lavis_paddle.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", help="path to configuration file.",default="/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/blip2/lavis_paddle/projects/blip2/train/pretrain_stage2.yaml")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed
    # seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    paddle.seed(seed)

    # cudnn.benchmark = False
    # cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    # init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    # paddle.device.set_device("cpu")
    model = task.build_model(cfg)
    # model.to("gpu")
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

# job_id = now()

# cfg = Config(parse_args())

# # init_distributed_mode(cfg.run_cfg)

# setup_seeds(cfg)

# # set after init_distributed_mode() to only log on master.
# setup_logger()

# cfg.pretty_print()

# task = tasks.setup_task(cfg)
# datasets = task.build_datasets(cfg)
# model_paddle = task.build_model(cfg)
if __name__ == "__main__":
    main()
