import paddle
import torch

#paddletotorch
# paddle_state_dict = paddle.load('blip2/lavis_paddle/output/BLIP2/Pretrain_stage2_paddle/20230619094/checkpoint_9.pdparams')['model']
# torch_init_dict = torch.load('checkpoint_9.pth')['model']
# new_state_dict = {}
# for i in torch_init_dict:
#     if i in paddle_state_dict:
#         torch_para = torch.as_tensor(paddle_state_dict[i].numpy())
#     else:
#         if 'opt_model.opt' not in i: 
#             print('name misatch: ', i)
#         continue
#     if torch_para.dtype == torch.float16:
#        torch_para= torch_para.astype(torch.float32)
#     if 'weight' in i and torch_para.shape == torch_init_dict[i].T.shape:
#         new_state_dict[i] = torch_para.T
#     elif torch_para.shape == torch_init_dict[i].shape:
#         new_state_dict[i] = torch_para
#     else:
#         print('weight mismatch: ', i, torch_init_dict[i].shape, paddle_state_dict[i].shape)

# torch.save(new_state_dict, 'checkpoint_9_paddle_20230619094.pth')
#torchtopaddle
import paddle
import torch


paddle_state_dict = paddle.load('blip2/blip2_stage2_pretrained.pdparams')
torch_init_dict = torch.load('/paddle/workspace/wjm/baidu/personal-code/PaddleNLP/blip2_caption_opt2.7b.pth')['model']
new_state_dict = {}
for i in paddle_state_dict:
    if 'qkv.bias' in i:
        name = i.split('.qkv')[0]
        q = torch_init_dict['{}.q_bias'.format(name)]
        v = torch_init_dict['{}.v_bias'.format(name)]
        qkv_bias = torch.cat((q, torch.zeros_like(v), v))
        paddle_para = paddle.to_tensor(qkv_bias.numpy())
    elif i in torch_init_dict:
        paddle_para = paddle.to_tensor(torch_init_dict[i].cpu().numpy())
    else:
        if 'opt_model.opt' not in i:
            print('name misatch: ', i)
        continue
    if paddle_para.dtype == paddle.float16:
        paddle_para = paddle_para.astype(paddle.float32)
    if 'weight' in i and paddle_para.shape == paddle_state_dict[i].T.shape:
        new_state_dict[i] = paddle_para.T
    elif paddle_para.shape == paddle_state_dict[i].shape:
        new_state_dict[i] = paddle_para
    else:
        print('weight mismatch: ', i, torch_init_dict[i].shape, paddle_state_dict[i].shape)

paddle.save(new_state_dict, 'blip2_caption_opt2.7b.pdparams')