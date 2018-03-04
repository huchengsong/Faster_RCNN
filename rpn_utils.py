import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from box_parametrize import box_parameterize
from bbox_IoU import bbox_IoU


def generate_training_anchors(roi, gt_bbox, gt_label,
                              num_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5,
                              neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
                              loc_normalize_mean=(0., 0., 0., 0.),
                              loc_normalize_std=(0.1, 0.1, 0.2, 0.2),):

    """
    generate ground truth label and location for sampled proposals
    :param roi: (N, 4) ndarray; region of interest from rpn proposal
    :param gt_bbox: (K, 4) ndarray; ground truth bounding box an image
    :param gt_label: (K, ) ndarray; ground truth label for each bounding box; range[1, class_number]
    :param num_sample: number of sampled roi
    :param pos_ratio: ratio of positive samples in the output
    :param pos_iou_thresh: positive iou threshold
    :param neg_iou_thresh_hi: negative iou threshold low end
    :param neg_iou_thresh_lo: negative iou threshold high end
    :param loc_normalize_mean: mean
    :param loc_normalize_std: standard deviation
    :return: ndarray: sampled_roi, gt_loc, gt_label
    """
    iou_matrix = bbox_IoU(gt_bbox, roi)
    roi_gt_assignment = iou_matrix.argmax(axis=0)
    max_iou = iou_matrix.max(axis=0)
    gt_roi_label = gt_label[roi_gt_assignment]

    # sample positive roi and get roi index
    max_num_pos_roi = int(pos_ratio * num_sample)
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    num_pos_roi = int(min(max_num_pos_roi, len(pos_index)))
    if num_pos_roi > 0:
        pos_index = np.random.choice(pos_index, num_pos_roi, replace=False)

    # sample negative roi and get roi index
    neg_index = np.where((max_iou < neg_iou_thresh_hi) &
                         (max_iou >= neg_iou_thresh_lo))[0]
    max_num_neg_roi = num_sample - num_pos_roi
    num_neg_roi = int(min(max_num_neg_roi, len(neg_index)))
    if num_neg_roi > 0:
        neg_index = np.random.choice(neg_index, num_neg_roi, replace=False)

    # get sampled rois and their labels
    keep_index = np.append(pos_index, neg_index)
    gt_roi_label = gt_roi_label[keep_index]
    gt_roi_label[num_pos_roi:] = 0
    sampled_roi = roi[keep_index]

    # get parameterized roi
    print(len(keep_index))
    gt_roi_loc = box_parameterize(gt_bbox[roi_gt_assignment[keep_index]], sampled_roi)
    gt_roi_loc = (gt_roi_loc - loc_normalize_mean) / loc_normalize_std

    return sampled_roi, gt_roi_loc, gt_roi_label


def generate_anchor_loc_label(anchor, gt_bbox, img_size,
                              num_sample=256, pos_iou_thresh=0.7,
                              neg_iou_thresh=0.3, pos_ratio=0.5):

    num_anchors = anchor.shape[0]

    # cross-boundary anchors will not be used
    ind_inside_img = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= img_size[0]) &  # height
        (anchor[:, 3] <= img_size[1])  # width
    )[0]

    selected_anchors = anchor[ind_inside_img, :]
    labels = np.empty(selected_anchors.shape[0], dtype=np.int32)
    labels.fill(-1)

    iou_matrix = bbox_IoU(gt_bbox, selected_anchors)
    ind_max_each_anchor = iou_matrix.argmax(axis=0)
    max_iou_each_anchor = iou_matrix.max(axis=0)

    # set 1 to positive anchors
    labels[max_iou_each_anchor >= pos_iou_thresh] = 1
    # set 0 to negative anchors
    labels[max_iou_each_anchor <= neg_iou_thresh] = 0
    # set 1 to the anchors that have the highest IoU with a certain ground-truth box
    max_iou_with_gt = iou_matrix.max(axis=1)
    ind_max_iou_with_gt = np.where(np.transpose(iou_matrix) == max_iou_with_gt)[0]
    labels[ind_max_iou_with_gt] = 1

    # if positive anchors are too many, reduce the positive anchor number
    num_pos_sample = int(pos_ratio * num_sample)
    ind_positive_anchor = np.array(np.where(labels == 1)).flatten()
    num_positive_anchor = ind_positive_anchor.size
    if num_positive_anchor > num_pos_sample:
        disable_inds = np.random.choice(
            ind_positive_anchor,
            size=int(num_positive_anchor - num_pos_sample),
            replace=False)
        labels[disable_inds] = -1

    # if negative anchors are too many, reduce the negative anchor number
    # if positive anchors are not enough, pad with negative anchors
    num_neg_sample = num_sample - np.sum(labels == 1)
    ind_negative_anchor = np.array(np.where(labels == 0)).flatten()
    num_negative_anchor = ind_negative_anchor.size
    if num_negative_anchor > num_neg_sample:
        disable_inds = np.random.choice(
            ind_negative_anchor,
            size=int(num_negative_anchor - num_neg_sample),
            replace=False)
        labels[disable_inds] = -1
    print(np.sum(labels == 1), np.sum(labels == 0))

    gt_box_parameterized = box_parameterize(
        gt_bbox[ind_max_each_anchor, :], selected_anchors)

    labels = _unmap(labels, num_anchors, ind_inside_img, fill=-1)
    gt_box_parameterized = _unmap(gt_box_parameterized, num_anchors, ind_inside_img, fill=0)

    return labels, gt_box_parameterized


def _unmap(data, num_original_data, index_of_data, fill=-1):
    """
    :param data: data to be unmaped to original size
    :param num_original_data: original_matrix.shape[0]
    :param index_of_data: index of data in original matrix
    :param fill: number to be filled in unmaped matrix
    :return: an unmaped matrix
    """
    if len(data.shape) == 1:
        ret = np.empty(num_original_data, dtype=data.dtype)
        ret.fill(fill)
        ret[index_of_data] = data
    else:
        ret_shape = np.array(data.shape)
        ret_shape[0] = num_original_data
        ret = np.empty(ret_shape, dtype=data.dtype)
        ret.fill(fill)
        ret[index_of_data, :] = data
    return ret


def rpn_loss(rpn_score, rpn_loc, gt_rpn_loc, gt_rpn_label, rpn_sigma):
    # rpn loc loss
    mask = Variable(torch.zeros(gt_rpn_loc.size())).cuda()
    mask[(gt_rpn_label > 0).view(-1, 1).expand_as(mask).cuda()] = 1
    loc_loss = _smooth_l1_loss(rpn_loc, gt_rpn_loc, mask, rpn_sigma)

    # normalize by the number of positive rois
    loc_loss /= (gt_rpn_label > 0).float().sum()

    # rpn cls loss
    # nn.CrossEntropy includes LogSoftMax and NLLLoss in one single function
    # TODO: check whether need normalization
    cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label)

    return cls_loss, loc_loss


def _smooth_l1_loss(x, gt, mask, sigma):
    """
    retrun smooth l1 loss
    :param x: [N, K], troch Variable
    :param gt: [N, K], troch Variable
    :param mask: [N, K], troch Variable
    :param sigma: constant
    :return: loss
    """
    sigma2 = sigma ** 2
    diff = mask * (x - gt)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    loss = y.sum()
    return loss


def test():
    # test generate_training_anchors(roi, gt_bbox, gt_label)
    roi = np.array([[0, 0, 100, 100],
                    [0, 0, 110, 110],
                    [0, 0, 125, 125],
                    [0, 0, 150, 150],
                    [500, 500, 600, 600],
                    [500, 500, 650, 650]])
    gt_bbox = np.array([[0, 0, 105, 105],
                       [500, 500, 600, 600]])
    gt_label = np.array([1, 2])
    sampled_roi, gt_roi_loc, gt_label = generate_training_anchors(roi, gt_bbox, gt_label, num_sample=6, pos_ratio=0.5)
    print('sampled_roi', sampled_roi)
    print('gt_roi_loc', (gt_roi_loc*[0.1, 0.1, 0.2, 0.2])+[0, 0, 0, 0])
    print('gt_label', gt_label)

    # test generate_anchor_loc_label()
    anchor = np.array([[0, 0, 10, 10],
                       [0, 0, 100, 100],
                       [0, 0, 110, 110],
                       [0, 0, 125, 125],
                       [0, 0, 150, 150],
                       [500, 500, 600, 600],
                       [500, 500, 650, 650]])
    gt_bbox = np.array([[0, 0, 105, 105],
                        [500, 500, 600, 600]])
    img_size = [600, 600]
    gt_label, gt_loc = generate_anchor_loc_label(anchor, gt_bbox, img_size, num_sample=6)
    print('gt_label', gt_label)
    print('gt_loc', gt_loc)

    # _smooth_l1_loss()
    x = Variable(torch.FloatTensor([0.5, 2]))
    gt = Variable(torch.FloatTensor([0, 0]))
    weight = Variable(torch.FloatTensor([1, 1]))
    sigma = 1
    loss = _smooth_l1_loss(x, gt, weight, sigma)
    print(loss)

    # TODO: test rpn_loss()


if __name__ == "__main__":
    test()
