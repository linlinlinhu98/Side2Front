# coding: utf-8
# Pure Python NMS implementation (no Cython required)
import numpy as np


def cpu_nms(dets, thresh):
    """Pure Python implementation of Non-Maximum Suppression."""
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def cpu_soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    """Pure Python implementation of Soft-NMS."""
    N = boxes.shape[0]
    for i in range(N):
        maxpos = i
        maxscore = boxes[i, 4]

        for pos in range(i + 1, N):
            if boxes[pos, 4] > maxscore:
                maxscore = boxes[pos, 4]
                maxpos = pos

        boxes[[i, maxpos]] = boxes[[maxpos, i]]

        tx1, ty1, tx2, ty2, ts = boxes[i]

        for pos in range(i + 1, N):
            xx1 = max(tx1, boxes[pos, 0])
            yy1 = max(ty1, boxes[pos, 1])
            xx2 = min(tx2, boxes[pos, 2])
            yy2 = min(ty2, boxes[pos, 3])

            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h

            area = (boxes[pos, 2] - boxes[pos, 0] + 1) * (boxes[pos, 3] - boxes[pos, 1] + 1)
            ua = (tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - inter
            ovr = inter / ua

            if method == 1:
                weight = 1 - ovr if ovr > Nt else 1
            elif method == 2:
                weight = np.exp(-(ovr * ovr) / sigma)
            else:
                weight = 0 if ovr > Nt else 1

            boxes[pos, 4] *= weight

    keep = [i for i in range(N) if boxes[i, 4] >= threshold]
    return keep
