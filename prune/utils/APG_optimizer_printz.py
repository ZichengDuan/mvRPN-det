#!/usr/bin/env python
# -*- coding-utf-8 -*-
# xuer ----time:

import torch
from torch.optim.optimizer import Optimizer


class APGNAG_printz(Optimizer):
    def __init__(self, params, device, lr=None, momentum=0, dampening=0,
                 weight_decay=0, gamma=0.01, nesterov=False):
        self.gamma = gamma
        self.device = device
        if lr is None:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(APGNAG_printz, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(APGNAG_printz, self).__setstate__(state)
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
            dampening = group['dampening']  # 回溯
            nesterov = group['nesterov']

            for p in group['params']:
                if torch.numel(p) == 1:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    # if self.gamma != 0:
                    #     d_p.add_(2 * group['lr'] * self.gamma, torch.sign(p.data))
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            z = p.data - group['lr'] * d_p.data  # z = p - lr * dp
                            z = self.soft_thresholding(z, group['lr'] * self.gamma)  # z = prox(z) 正则化的解
                            buf.mul_(momentum).add_(z - p.data)  # buf = z- p + momentum * buf
                            p.data = z  # p = z + momentum * buf
                        else:
                            p.data = p.data + momentum * buf
                            buf = param_state['momentum_buffer']
                            z = p.data - group['lr'] * d_p.data  # z = p - lr * dp
                            z = self.soft_thresholding(z, group['lr'] * self.gamma)  # z = prox(z)
                            buf.mul_(momentum).add_(z - p.data)  # buf = z- p + momentum * buf
                            p.data = z  # p = prox(z) + momentum * buf
                        a = torch.FloatTensor([0.0, p.data])
                        p.data = torch.max(a).to(self.device)
                        print(p.data)
                        # p = max{(prox(p - lr * dp) + momentum * (prox(p - lr * dp) - p + (momentum * buf)), 0}
                        # 即 p = max{(prox(p - lr * dp) + momentum * momentum * buf + momentum * (prox(p - lr * dp) - p), 0}
                        # 如没有prox, 即 p = max{(p - lr * dp + momentum * (-lr * dp + (momentum * buf)), 0}
                        # 即 p = max{(p - (1 + momentum) * lr * dp + momentum * momentum * buf), 0}
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
                            d_p = d_p.add(momentum, buf)  # d_p = d_p + momentum * buf  # 把独立计算梯度部分简化了
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)  # p = p - lr * d_p
                    # p = p - lr * (buf * momentum + d_p)
                    # p = p - lr * (momentum * buf + (1 - dampening) * d_p)
                    # if nesterov: p = p - lr * (d_p + momentum * (momentum * buf + (1 - dampening) * d_p))
                    # 即 p = p - lr * momentum * momentum * buf - lr * (1 + momentum * (1 - dampening))* d_p
        return loss
