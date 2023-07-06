# """
#  Copyright (c) 2022, salesforce.com, inc.
#  All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#  For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# """

# import datetime
# import functools
# import os
# import logging

# import paddle
# import paddle.distributed as dist
# from paddle.distributed import init_parallel_env
# #import timm.models.hub as timm_hub


# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__

#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop("force", False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()


# def is_main_process():
#     return get_rank() == 0


# def init_distributed_mode(args):
#     #if int(os.environ.get("PADDLE_TRAINERS_NUM", 1)) > 1:
#     if True:
#         args.distributed = True

#         paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
#         init_parallel_env()
#         paddle.distributed.barrier()
#         #paddle.set_device(args.gpu)
#         args.dist_backend = "nccl"
#         print(
#             "| distributed init (rank {}, world {}): {}".format(
#                 int(os.environ.get("PADDLE_RANK_IN_NODE", 0)), 
#                 int(os.environ.get("PADDLE_TRAINERS_NUM", 1)), 
#                 args.dist_url
#             ))
#     else:
#         args.distributed = False
#     #setup_for_distributed(args.rank == 0)

# def init_distributed_mode_bp(args):
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ["WORLD_SIZE"])
#         args.gpu = int(os.environ["LOCAL_RANK"])
#     elif "SLURM_PROCID" in os.environ:
#         args.rank = int(os.environ["SLURM_PROCID"])
#         args.gpu = args.rank % paddle.cuda.device_count()
#     else:
#         print("Not using distributed mode")
#         args.distributed = False
#         return

#     args.distributed = True

#     # paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
#     # init_parallel_env()
#     # paddle.distributed.barrier()
#     paddle.set_device(args.gpu)
#     args.dist_backend = "nccl"
#     logging.info(
#         "| distributed init (rank {}, world {}): {}".format(
#             args.rank, args.world_size, args.dist_url
#         ),
#         flush=True,
#     )
    
#     dist.init_process_group(
#         backend=args.dist_backend,
#         init_method=args.dist_url,
#         world_size=args.world_size,
#         rank=args.rank,
#         timeout=datetime.timedelta(
#             days=365
#         ),  # allow auto-downloading and de-compressing
#     )
#     dist.barrier()
#     setup_for_distributed(args.rank == 0)




