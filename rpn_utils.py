import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from box_parametrize import box_parameterize
from generate_anchors import generate_anchors
from bbox_IoU import bbox_IoU

np.set_printoptions(threshold=np.inf)
BATCH_SIZE = 256


def generate_gt_cls_reg(img_info, score_dim, base_size, ratios, scales):
    """
    :param img_info: dictionary with key ('img_size', 'objects')
    :param cls_score_dim: (H, W) of the class score matrix
    :param base_size: stride size of anchor on an image
    :param scales: scales of anchors
    :return: labels: (1, 2 * num_ratio * num_scale, H, W)
    :return: gt_box_parameterized: (1, 4 * num_ratio * num_scale, H, W)
    """
    ground_truth_boxes = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
    anchor_list = generate_anchors(base_size=base_size, ratios=ratios, scales=scales)
    score_height, score_width = int(score_dim[0]), int(score_dim[1])

    anchor_x_shift = np.arange(0, score_width) *\
                     base_size + base_size/2
    anchor_y_shift = np.arange(0, score_height) *\
                     base_size + base_size/2

    anchor_centers = np.array([[i, j]
                               for i in anchor_y_shift
                               for j in anchor_x_shift])
    all_anchors = np.array([np.tile(anchor_centers[i], 2) + anchor_list[j]
                            for i in range(0, anchor_centers.shape[0])
                            for j in range(0, anchor_list.shape[0])])

    num_all_anchors = all_anchors.shape[0]

    # cross-boundary anchors will not be used
    inds_inside_img = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] < img_info['img_size'][0]) &  # height
        (all_anchors[:, 3] < img_info['img_size'][1])  # width
    )[0]
    selected_anchors = all_anchors[inds_inside_img, :]
    
    labels = np.zeros(selected_anchors.shape[0], dtype=np.float32)
    iou_matrix = bbox_IoU(ground_truth_boxes, selected_anchors)  # (N, K) ndarray

    # set 1 to the anchors that have IoU larger than 0.7
    ind_max_each_anchor = np.argmax(iou_matrix, axis=0)
    max_iou_each_anchor = iou_matrix[ind_max_each_anchor, np.arange(selected_anchors.shape[0])]
    labels[max_iou_each_anchor >= 0.7] = 1
    # set -1 to the anchors that have IoU larger than 0.3
    labels[max_iou_each_anchor <= 0.3] = -1

    # set 1 to the anchors that have the highest IoU with a ground-truth box
    # overwrite -1s for the anchors that have <0.3 IoU but have the highest IoU for a gt box
    ind_max_iou_with_gt = np.argmax(iou_matrix, axis=1)
    max_iou_with_gt = iou_matrix[np.arange(ground_truth_boxes.shape[0]), ind_max_iou_with_gt]
    ind_max_iou_with_gt = np.where(np.transpose(iou_matrix) == max_iou_with_gt)[0]
    labels[ind_max_iou_with_gt] = 1

    # batch size 256. ratio of positive and negative anchors 1:1
    # if positive anchors are too many, reduce the positive anchor number
    ind_positive_anchor = np.array(np.where(labels == 1)).flatten()
    num_positive_anchor = ind_positive_anchor.size
    if num_positive_anchor > BATCH_SIZE / 2:
        disable_inds = np.random.choice(
            ind_positive_anchor,
            size=int(num_positive_anchor - BATCH_SIZE / 2),
            replace=False)
        labels[disable_inds] = 0

    # if negative anchors are too many, reduce the negative anchor number
    # if positive anchors are not enough, pad with negative anchors
    ind_negative_anchor = np.array(np.where(labels == -1)).flatten()
    num_negative_anchor = ind_negative_anchor.size
    if num_negative_anchor > max(BATCH_SIZE - num_positive_anchor, BATCH_SIZE / 2):
        disable_inds = np.random.choice(
            ind_negative_anchor,
            size=int(num_negative_anchor - max(BATCH_SIZE - num_positive_anchor, BATCH_SIZE / 2)),
            replace=False)
        labels[disable_inds] = 0
    print(np.sum(labels == 1), np.sum(labels == -1))
    gt_box_parameterized = box_parameterize(
        ground_truth_boxes[ind_max_each_anchor, :], selected_anchors)

    labels = _unmap(labels, num_all_anchors, inds_inside_img, fill=0)
    labels = np.reshape(
        labels,
        [1, score_height, score_width, anchor_list.shape[0]],
        order='C')
    labels = np.transpose(labels, (0, 3, 1, 2))
    one_hot_labels = np.repeat(labels, 2, axis=1)
    one_hot_labels[:, [2 * k for k in range(anchor_list.shape[0])], :, :] = \
        one_hot_labels[:, [2 * k for k in range(anchor_list.shape[0])], :, :] == 1
    one_hot_labels[:, [2 * k + 1 for k in range(anchor_list.shape[0])], :, :] = \
        one_hot_labels[:, [2 * k +1 for k in range(anchor_list.shape[0])], :, :] == -1

    gt_box_parameterized = _unmap(gt_box_parameterized, num_all_anchors, inds_inside_img, fill=0)
    gt_box_parameterized = np.reshape(
        gt_box_parameterized,
        [1, score_height, score_width, 4*anchor_list.shape[0]],
        order='C')
    gt_box_parameterized = np.transpose(gt_box_parameterized, (0, 3, 1, 2))

    return one_hot_labels, gt_box_parameterized


def _unmap(data, num_original_data, index_of_data, fill=0):
    """
    :param data: data to be unmaped to original size
    :param num_original_data: original_matrix.shape[0]
    :param index_of_data: index of data in original matrix
    :param fill: number to be filled in unmaped matrix
    :return: an unmaped matrix
    """
    if len(data.shape) == 1:
        ret = np.empty(num_original_data, dtype=np.float32)
        ret.fill(fill)
        ret[index_of_data] = data
    else:
        ret_shape = np.array(data.shape)
        ret_shape[0] = num_original_data
        ret = np.empty(ret_shape, dtype=np.float32)
        ret.fill(fill)
        ret[index_of_data, :] = data
    return ret


def generate_rpn_loss(cls_score, reg_score, cls_gt, reg_gt, cuda=False):
    """
    return loss of region proposal networks
    :param cls_score: (1, 2 * num_ratio * num_scale, H, W), predicted class scores, pytorch Variable
    :param reg_score: (1, 4 * num_ratio * num_scale, H, W), predicted bounding box scores, pytorch Variable
    :param cls_gt: (1, 2 * num_ratio * num_scale, H, W), class ground truth, np array
    :param reg_gt: (1, 4 * num_ratio * num_scale, H, W), bounding box ground truth, np array
    :return: region proposal network loss
    """
    num_anchors = int(cls_score.size()[1] / 2)
    softmax = nn.Softmin(dim=1)
    for i in range(0, num_anchors):
        cls_score[:, (2 * i, 2 * i + 1), :, :] = \
            softmax(cls_score[:, (2 * i, 2 * i + 1), :, :])
    # clamp softmax result to avoid numeric error
    cls_score = torch.clamp(cls_score, min=1e-12, max=1-1e-12)
    minus_log_cls_score = torch.mul(torch.log(cls_score), -1)

    # calculate cross-entropy loss of class scores
    n_cls = Variable(torch.FloatTensor(np.array([np.sum(cls_gt == 1)])))
    if cuda:
        n_cls = n_cls.cuda()
    cls_gt_tensor = Variable(torch.from_numpy(cls_gt).type(torch.FloatTensor))
    if cuda:
        cls_gt_tensor = cls_gt_tensor.cuda()
    cls_loss = torch.div(
        torch.sum(torch.mul(minus_log_cls_score, cls_gt_tensor)),
        n_cls)

    # calculate regression loss for bounding box prediction
    # smooth L1 loss
    n_reg = Variable(torch.FloatTensor(np.array([(cls_score.size()[2] * cls_score.size()[3])])))
    if cuda:
        n_reg = n_reg.cuda()
    reg_gt = Variable(torch.from_numpy(reg_gt))
    if cuda:
        reg_gt = reg_gt.cuda()
    diff_reg = reg_score - reg_gt
    abs_diff_reg = torch.abs(diff_reg)
    if not cuda:
        loss_larger_than_one = torch.mul((abs_diff_reg >= 1).type(torch.FloatTensor),
                                         abs_diff_reg - 0.5)
        loss_smaller_than_one = torch.mul((abs_diff_reg < 1).type(torch.FloatTensor),
                                          torch.mul(torch.pow(abs_diff_reg, 2), 0.5))
    if cuda:
        loss_larger_than_one = torch.mul((abs_diff_reg >= 1).type(torch.FloatTensor).cuda(),
                                         abs_diff_reg - 0.5)
        loss_smaller_than_one = torch.mul((abs_diff_reg < 1).type(torch.FloatTensor).cuda(),
                                          torch.mul(torch.pow(abs_diff_reg, 2), 0.5))
    sum_reg_loss = torch.add(loss_larger_than_one, loss_smaller_than_one)
    reg_mask = np.repeat(cls_gt[:, [2 * k for k in range(num_anchors)], :, :], 4, axis=1)
    reg_mask = Variable(torch.from_numpy(reg_mask).type(torch.FloatTensor))
    if cuda:
        reg_mask = reg_mask.cuda()
    sum_reg_loss = torch.sum(torch.mul(reg_mask, sum_reg_loss))
    reg_loss = torch.mul(torch.div(sum_reg_loss, n_reg), 10)
    return torch.add(cls_loss, reg_loss)


if __name__ == "__main__":
    import cv2
    from rescale_image import rescale_image

    # # test _unmap
    # data = np.array([[1], [2], [3]])
    # num_original_data = 10
    # index_of_data = np.array([3, 7, 9])
    # fill = -1
    # ret = _unmap(data, num_original_data, index_of_data, fill)
    # print(ret)
    # test generate_gt_cls_reg
    img_box_dict = np.load('../VOCdevkit/img_box_dict.npy')[()]
    for img_dir, img_info in img_box_dict.items():
        image, modified_image_info = rescale_image(img_dir, img_info)
        cls_score_dim = np.array([np.floor(modified_image_info['img_size'][0]/16).astype(np.int),
                              np.floor(modified_image_info['img_size'][1]/16).astype(np.int)])
        base_size = 16
        ratios = [0.5, 1.0, 2.0]
        scales = [8, 16, 32]
        one_hot_label, gt_box = generate_gt_cls_reg(img_info, cls_score_dim, base_size, ratios, scales)
        # draw ground truth boxes on image
        for object in modified_image_info['objects']:
            ymin, xmin, ymax, xmax = [int(i) for i in object[1:5]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(image,
                    object[0],
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

        # generate anchors
        anchor_list = generate_anchors(base_size=base_size, ratios=ratios, scales=scales)
        anchor_x_shift = np.arange(0, cls_score_dim[1]) * \
                     base_size + base_size / 2
        anchor_y_shift = np.arange(0, cls_score_dim[0]) * \
                     base_size + base_size / 2
        anchor_centers = np.array([[i, j]
                               for i in anchor_y_shift
                               for j in anchor_x_shift])
        all_anchors = np.array([np.tile(anchor_centers[i], 2) + anchor_list[j]
                            for i in range(0, anchor_centers.shape[0])
                            for j in range(0, anchor_list.shape[0])])
        all_anchors = np.reshape(
            all_anchors,
            [1, cls_score_dim[0], cls_score_dim[1], 4 * anchor_list.shape[0]],
            order='C')
        all_anchors = np.transpose(all_anchors, (0, 3, 1, 2))
        # print('positive: ', np.sum(label == 1))
        # print('zero', np.sum(label == 0))
        # print('negative', np.sum(label == -1))
        # print(all_anchors.shape)

        # draw positive anchors
        for i in range(cls_score_dim[0]):
            for j in range(cls_score_dim[1]):
                for k in range(len(ratios)*len(scales)):
                    if one_hot_label[0, 2 * k, i, j] == 1:
                        anchor = all_anchors[0, k*4:(k+1)*4, i, j]
                        ymin, xmin, ymax, xmax = [int(i) for i in anchor]
                        color = np.squeeze([np.random.randint(255, size=1),
                                            np.random.randint(255, size=1),
                                            np.random.randint(255, size=1)]).tolist()
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cls_score = Variable(torch.randn((1, 4, 40, 40)))
    reg_score = Variable(torch.randn((1, 8, 40, 40)))
    cls_gt = np.random.randint(0, 2, size=6400).reshape((1, 4, 40, 40))
    reg_gt = torch.randn((1, 8, 40, 40)).numpy()
    loss = generate_rpn_loss(cls_score, reg_score, cls_gt, reg_gt)
    print(loss.data[0])