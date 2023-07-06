Blip2-Paddle
===
paddlebilp2 stage2

快速开始
---
* 安装

    安装 paddlepaddle-gpu 2.4.2

    安装 paddlenlp-2.5.2.post0, 版本:  f9d9fd7176139741474f820e367c01988019d200

    更多的参见 `requirements.txt`

* 训练
```shell
 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' sh run_scripts/blip2/train/pretrain_stage1.sh 
```
* 测试

```shell
 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' sh run_scripts/blip2/eval/pretrain_stage1.sh 
```

如何贡献
---
贡献patch流程及质量要求

版本信息
---
本项目的各版本信息和变更历史可以在[这里][changelog]查看。

维护者
---
### owners
* sunting13(sunting13@baidu.com)

### committers
* sunting13(sunting13@baidu.com)

讨论
---
百度Hi交流群：群号


[changelog]: http://icode.baidu.com/repos/baidu/personal-code/blip2/blob/master:CHANGELOG.md
