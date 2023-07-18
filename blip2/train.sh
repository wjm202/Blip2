CUDA_VISIBLE_DEVICES=4,5,6,7 python -m paddle.distributed.launch train_paddle.py --cfg-path lavis_paddle/projects/blip2/train/pretrain_stage2.yaml
