"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math
from paddle.optimizer.lr import LRScheduler, CosineAnnealingDecay, LinearWarmup

from lavis_paddle.common.registry import registry


@registry.register_lr_scheduler("linear_warmup_cosine_lr_paddle")
class Cosine(LRScheduler):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        eta_min(float): Minimum learning rate. Default: 0.0.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 epochs,
                 eta_min=0.0,
                 warmup_steps=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        self.start_lr = learning_rate
        self.T_max = epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.last_lr = self.start_lr
        self.cur_step = 0
        if self.warmup_steps > 0:
            self.last_lr = self.warmup_start_lr
        super().__init__(learning_rate=self.last_lr, last_epoch=self.last_epoch)

    def step(self, cur_epoch=0, cur_step=0):
        self.cur_step += 1
        if self.cur_step < self.warmup_steps and cur_epoch == 0:
            self.last_lr = self.warmup_start_lr + (self.start_lr - self.warmup_start_lr) *\
                           cur_step / max(self.warmup_steps, 1)
        else:
            self.last_lr = (self.start_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * cur_epoch / self.T_max)
    ) + self.eta_min
        self.last_epoch = cur_epoch

    def get_lr(self):
        return self.last_lr
