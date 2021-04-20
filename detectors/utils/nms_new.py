import torch
import numpy as np


def nms_new(bboxes, confidence, left=4, threshold=0.1):
    bbox = bboxes.squeeze().astype(int)
    confidence = confidence.squeeze()
    keep = torch.zeros(confidence.shape).long()
    if len(bbox) == 0:
        return keep

    v, indices = confidence.sort(0)  # sort in ascending order

    bbox_keep = []
    indices_keep = []
    i = 1
    while len(bbox_keep) < left and len(indices) > 0:
        if len(bbox_keep) == 0:
            bbox_keep.append(bbox[indices[-1]])
        elif keep_box(bbox_keep, bbox[indices[-1]], iou_threash=threshold):
            bbox_keep.append(bbox[indices[-1]])
            indices_keep.append((indices[-1]).item())
        # confidence = confidence[:-1]
        indices = indices[:-1]
        i += 1
        # print(i, len(indices))
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
        res = res and (bbox_iou(box, target)[0][0] < iou_threash)
        if not res:
            return res
    return res
    pass
