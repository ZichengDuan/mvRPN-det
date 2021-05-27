#!/usr/bin/env python
# -*- coding-utf-8 -*-
# xuer ----time:

import torch
from torch.optim.optimizer import Optimizer


class APGNAG_prox(Optimizer):
    def __init__(self, params, lr=None, momentum=0, dampening=0,
                 weight_decay=0, gamma=0.01, nesterov=False, nesterovp=True):
        self.gamma = gamma
        # 以原版SGD为基础修改,不需要加转移device
        if lr is None:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, nesterovp=nesterovp)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(APGNAG_prox, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(APGNAG_prox, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @staticmethod
    def soft_thresholding(input, alpha):
        a = torch.FloatTensor([0.0, torch.abs(input) - alpha])
        return torch.sign(input) * torch.max(a)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            nesterovp = group['nesterovp']

            for p in group['params']:
                if torch.numel(p) == 1:
                    if p.grad is None:
                        continue
                    # d_p = p.grad.data
                    d_p = p.grad.data.add_(self.gamma * torch.sign(p.grad.data))
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)  # buf = buf * momentum + d_p
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)  # buf = momentum * buf + (1 - dampening) * d_p
                        if nesterovp:
                            d_p = d_p.add(momentum, buf)  # d_p = d_p + momentum * buf
                        else:
                            d_p = buf
                    p.data.add_(-group['lr'], d_p)  # p = p - lr * d_p
                    # p.data = self.soft_thresholding(p.data, group['lr'] * self.gamma)
                    # p.data = self.soft_thresholding(p.data, self.gamma)
                else:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)  # d_p = d_p + (weight_decay * p)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)  # buf = buf * momentum + d_p
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)  # buf = momentum * buf + (1 - dampening) * d_p
                        if nesterov:
                            d_p = d_p.add(momentum, buf)  # d_p = d_p + momentum * buf
                        else:
                            d_p = buf
                    p.data.add_(-group['lr'], d_p)  # p = p - lr * d_p
                    # p = p - lr * (buf * momentum + d_p)
                    # p = p - lr * (momentum * buf + (1 - dampening) * d_p)
                    # if nesterov: p = p - lr * (d_p + momentum * (momentum * buf + (1 - dampening) * d_p))
                    # 即 p = p - lr * momentum * momentum * buf - lr * (1 + momentum * (1 - dampening))* d_p
        return loss
