import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torch as t
from torch import nn

from .utils.bbox_tools import generate_anchor_base
from .utils.creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=1026, mid_channels=1026, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=4,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # 生成anchors
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # torchvision

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # print(x.shape, img_size)
        # batch_images = torch.zeros((1, 3, img_size[0], img_size[1]))
        # image_sizes = [(img_size[0], img_size[1])]
        # image_list_ = image_list.ImageList(batch_images, image_sizes)
        # test_anchors = self.anchor_generator(image_list_, x)
        # print(test_anchors[0][:20])

        # a = np.zeros((img_size[0] + 100, img_size[1] + 100))
        # img = np.uint8(a)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # for anchor in test_anchors[0][:5]:
        #     x1, y1, x2, y2 = anchor
        #     cv2.rectangle(img, (int(x1+ 50), int(y1+ 50)), (int(x2+ 50), int(y2+ 50)), color=(255, 255, 0))
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/test_anchor.jpg", img)

        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        # a = np.zeros((img_size[0]+ 100, img_size[1]+ 100))
        # img = np.uint8(a)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #
        # for anchors in anchor[:5]:
        #     y1, x1, y2, x2 = anchors
        #     cv2.rectangle(img, (int(x2+ 50), int(y2+ 50)),(int(x1+ 50), int(y1+ 50)), color=(255, 255, 0))
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/anchor.jpg", img)

        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        # s = time.time()
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        # e = time.time()
        # print(e - s)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_locs, rpn_scores, anchor, rois, roi_indices


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)

    # print(shift_x)

    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
