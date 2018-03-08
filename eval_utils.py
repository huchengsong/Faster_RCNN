import numpy as np

def eval_detection_voc(bboxes, labels, scores, gt_bboxes, gt_labels,
                       iou_thresh=0.5, use_07_metric=False):
    """
    Calculate average precisions
    :param bboxes: ndarray (N, 4)
    :param labels: ndarray (N, )
    :param scores: ndarray (N, )
    :param gt_bboxes: ndarray (K, 4)
    :param gt_labels: ndarray (K, )
    :param iou_thresh: threshold
    :param use_07_metric: bool
    :return:
    """


def calc_precision_recall(bboxes, labels, scores, gt_bboxes, gt_labels,
                          iou_thresh=0.5, use_07_metric=False):
