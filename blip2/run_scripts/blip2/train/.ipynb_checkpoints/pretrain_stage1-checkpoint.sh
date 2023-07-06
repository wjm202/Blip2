#python train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml
python -m paddle.distributed.launch train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml
