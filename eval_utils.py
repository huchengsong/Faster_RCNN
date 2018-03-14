from collections import defaultdict
import numpy as np
from numba import jit

from bbox_IoU import bbox_IoU
from configure import Config


def calc_map(bboxes, labels, scores, gt_bboxes, gt_labels,
                       iou_thresh=0.5):
    """
    Calculate average precisions
    :param bboxes: list of N ndarray (K, 4)
    :param labels: list of N ndarray (K, )
    :param scores: list of N ndarray (K, )
    :param gt_bboxes: list of N ndarray (J, 4)
    :param gt_labels: list of N ndarray (J, )
    :param iou_thresh: threshold
    :return:
    """

    prec, rec = calc_precision_recall(bboxes, labels, scores, gt_bboxes, gt_labels, iou_thresh)
    ap = calc_ap(prec, rec)
    return ap, np.nanmean(ap)


@jit
def calc_precision_recall(bboxes, labels, scores, gt_bboxes,
                          gt_labels, iou_thresh=0.5,
                          num_class=Config.num_class):
    """
    return precision and recall for each class. N: number of images
    :param bboxes: list of N (K, 4) ndarray
    :param labels: list of N (K, ) ndarray
    :param scores: list of N  (K, ) ndarray
    :param gt_bboxes: list of N (J, 4) ndarray
    :param gt_labels: list of N (J, ) ndarray
    :param iou_thresh: threshold
    :param num_class: classes
    :return: prec, rec: list (num_class, ), with cumulative recall and precision for each class
    """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    # repeat over N images
    for i in range(len(bboxes)):
        boxes_i = bboxes[i]
        labels_i = labels[i]
        scores_i = scores[i]
        gt_boxes_i = gt_bboxes[i]
        gt_labels_i = gt_labels[i]

        # go through each class id in prediction and gt labels
        for l in np.unique(np.concatenate((gt_labels_i, labels_i))):
            pred_l = labels_i == l
            pred_box_l = boxes_i[pred_l]
            pred_score_l = scores_i[pred_l]

            # sort from largest score to smallest score
            order = pred_score_l.argsort()[::-1]
            pred_box_l = pred_box_l[order]
            pred_score_l = pred_score_l[order]

            # selected ground truth bounding box for class l
            gt_l = gt_labels_i == l
            gt_box_l = gt_boxes_i[gt_l]

            # add the number of gt bbox for class l
            n_pos[l] += (len(gt_box_l))

            # there is no prediction box for class l
            if len(pred_box_l) == 0:
                continue
            # there is no gt box for class l
            if len(gt_box_l) == 0:
                # record the score for class l
                score[l].extend(pred_score_l)
                match[l].extend([0] * len(pred_score_l))
                continue
            # record the score for class l
            score[l].extend(pred_score_l)
            # calculate iou
            iou = bbox_IoU(gt_box_l, pred_box_l)
            # assign index with iou smaller than iou_thresh to -1
            gt_index = iou.argmax(axis=0)
            gt_index[iou.max(axis=0) < iou_thresh] = -1

            select = np.zeros(len(gt_box_l), dtype=np.bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not select[gt_idx]:
                        match[l].append(1)
                    else:
                         match[l].append(0)
                    # if two bounding box are predicted
                    # for the same gt box, the second one
                    # will be assigned as 0
                    select[gt_idx] = True
                else:
                    match[l].append(0)

    prec = [None] * num_class
    rec = [None] * num_class
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        # return the cumulative sum of true positive and false positive
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_ap(prec, rec):
    num_classes = len(prec)
    ap = np.empty(num_classes)
    for i in range(num_classes):
        if prec[i] is None or rec[i] is None:
            ap[i] = np.nan
            continue
        mean_prec = np.concatenate(([[0], prec[i], [0]]))
        mean_rec = np.concatenate(([[0], rec[i], [1]]))
        mean_prec = np.maximum.accumulate(mean_prec[::-1])[::-1]
        index = np.where(mean_rec[1:] != mean_rec[:-1])[0]
        ap[i] = np.sum((mean_rec[index + 1] - mean_rec[index]) * mean_prec[index + 1])
    return ap


def test():
    boxes = []
    labels = []
    scores = [np.array([0.9, 0.5, 0.6]), np.array([0.6, 0.3])]
    b_1 = np.array([[0, 0, 100, 100],
                    [100, 100, 200, 200],
                    [100, 100, 230, 230]])
    boxes.append(b_1)
    b_2 = np.array([[0, 0, 150, 150],
                    [100, 100, 250, 250]])
    boxes.append(b_2)
    l_1 = np.array([1, 2, 3])
    labels.append(l_1)
    l_2 = np.array([1, 2])
    labels.append(l_2)
    gt_bboxes = [np.array([[0, 0, 110, 110]]),
                 np.array([[100, 100, 210, 210]])]
    gt_labels = [np.array([1]), np.array([2])]

    prec, rec = calc_precision_recall(boxes, labels, scores, gt_bboxes, gt_labels)
    ap, map = calc_map(boxes, labels, scores, gt_bboxes, gt_labels)
    print(prec, rec)
    print(ap, map)


if __name__ == "__main__":
    test()