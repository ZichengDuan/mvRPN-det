import torch
import numpy as np
import cupy as cp
import sys
sys.path.append("..")
from detectors.models.utils.nms.non_maximum_suppression import non_maximum_suppression
from EX_CONST import Const

def vis_nms(boxes, scores, iou_threshold):
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    print(boxes.shape, scores.shape)
    return torch.ops.torchvision.nms(boxes, scores.reshape(-1, 1).squeeze(), iou_threshold)


def _suppress(raw_cls_bbox, raw_prob):
    bbox = list()
    label = list()
    score = list()
    # skip cls_id = 0 because it is the background class
    for l in range(1, Const.roi_classes + 1):
        cls_bbox_l = raw_cls_bbox.reshape((-1, Const.roi_classes + 1, 4))[:, l, :]
        prob_l = raw_prob[:, l]
        mask = prob_l > 0.5
        cls_bbox_l = cls_bbox_l[mask]
        prob_l = prob_l[mask]
        keep = non_maximum_suppression(
            cp.array(cls_bbox_l), 0.1, prob_l)
        keep = cp.asnumpy(keep)
        bbox.append(cls_bbox_l[keep])
        # The labels are in [0, self.n_class - 2].
        label.append((l - 1) * np.ones((len(keep),)))
        score.append(prob_l[keep])
    bbox = np.concatenate(bbox, axis=0).astype(np.float32)
    label = np.concatenate(label, axis=0).astype(np.int32)
    score = np.concatenate(score, axis=0).astype(np.float32)
    return bbox, label, score


def nms_new(bboxes, confidence, sincos, position_mark, threshold=0.01, prob_threshold = 0.8):
    bbox = bboxes.squeeze()
    sincos = sincos.squeeze()
    confidence = torch.tensor(confidence)
    keep = torch.zeros(confidence.shape).long()
    if len(bbox) == 0:
        return keep

    v, indices = confidence.sort(0)  # sort in ascending order

    bbox_keep = []
    indices_keep = []
    sincos_keep = []
    position_mark_keep = []
    i = 0
    while len(indices) > 0:
        # print(keep_box(bbox_keep, bbox[indices[-1]], iou_threash=threshold))
        # print(v[-1], threshold)
        if len(bbox_keep) == 0:
            bbox_keep.append(bbox[indices[-1]])
            sincos_keep.append(sincos[indices[-1]])
            position_mark_keep.append(position_mark[indices[-1]])
        elif keep_box(bbox_keep, bbox[indices[-1]], iou_threash=threshold):
            if v[-1] < prob_threshold:
                return bbox_keep, confidence[indices_keep], sincos_keep, position_mark_keep
            bbox_keep.append(bbox[indices[-1]])
            indices_keep.append((indices[-1]).item())
            sincos_keep.append(sincos[indices[-1]])
            position_mark_keep.append(position_mark[indices[-1]])
        indices = indices[:-1]
        v = v[:-1]
        i += 1

    return bbox_keep, confidence[indices_keep], sincos_keep, position_mark_keep


def nms_new2(bboxes, confidence, threshold=0.00, prob_threshold = 0.7):
    bbox = bboxes.squeeze()
    confidence = torch.tensor(confidence)
    keep = torch.zeros(confidence.shape).long()
    if len(bbox) == 0:
        return keep

    v, indices = confidence.sort(0)  # sort in ascending order
    # print(v)
    bbox_keep = []
    indices_keep = []
    sincos_keep = []
    position_mark_keep = []
    i = 0
    while len(indices) > 0:
        # print(keep_box(bbox_keep, bbox[indices[-1]], iou_threash=threshold))
        # print(v[-1], threshold)
        if len(bbox_keep) == 0:
            bbox_keep.append(bbox[indices[-1]])
        elif keep_box(bbox_keep, bbox[indices[-1]], iou_threash=threshold):
            if v[-1] < prob_threshold:
                return bbox_keep, confidence[indices_keep]
            bbox_keep.append(bbox[indices[-1]])
            indices_keep.append((indices[-1]).item())
        indices = indices[:-1]
        v = v[:-1]
        i += 1

    return bbox_keep, confidence[indices_keep]


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    # print(bbox_a, bbox_b)

    bbox_a = np.array(bbox_a).reshape((1, 4))
    bbox_b = np.array(bbox_b).reshape((1, 4))

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def keep_box(boxes, target, iou_threash=0.4):
    res = True
    for box in boxes:
        res = res and (bbox_iou(box, target)[0][0] <= iou_threash)
        if not res:
            return res
    return res
