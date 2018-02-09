import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from bbox_IoU import bbox_IoU
from box_parametrize import box_parameterize

KEY = ['background', 'aeroplane', 'bicycle', 'bird',
       'boat', 'bottle', 'bus', 'car',
       'cat', 'chair', 'cow', 'diningtable',
       'dog', 'horse', 'motorbike', 'person',
       'pottedplant', 'sheep', 'sofa', 'train',
       'tvmonitor']


def generate_gt(img_info, region_proposals):
    """
    :param img_info: dictionary with key ('img_size', 'objects')
    :param region_proposals: (N, 4) numpy array, region proposal
    :return: label: (N, ) labels
    :return: gt_box_parameterized: (N, 4) numpy array, parametreized ground truth box based on
     region_proposals
    """
    label = np.zeros(region_proposals.shape[0]).astype(np.int)
    ground_truth_boxes = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
    ground_truth_class = np.array([KEY.index(cls) for cls in np.array(img_info['objects'])[:, 0]])

    # label each region proposal
    iou_matrix = bbox_IoU(ground_truth_boxes, region_proposals)
    ind_max_each_proposal = np.argmax(iou_matrix, axis=0)
    max_iou_each_proposal = iou_matrix[ind_max_each_proposal, np.arange(region_proposals.shape[0])]
    ind_positive = np.where(max_iou_each_proposal >= 0.5)
    label[ind_positive] = ground_truth_class[ind_max_each_proposal[ind_positive]]
    ind_negative = np.where(np.logical_and(
        max_iou_each_proposal < 0.5,
        max_iou_each_proposal >= 0.1))
    label[ind_negative] = 0

    # proposal parameterize based on ground truth
    gt_box_parameterized = box_parameterize(ground_truth_boxes[ind_max_each_proposal], region_proposals)
    gt_box_parameterized[np.where(label == 0), :] = 0

    return label, gt_box_parameterized


def generate_loss(class_pred, bbox_pred, label_gt, bbox_gt, cuda=False):
    """
    calculate loss of fast R-CNN
    :param class_pred: (N, class_num), pytorch Variable, class prediction
    :param bbox_pred: (N, 4 * class_num), pytorch Variable, bounding box prediction
    :param label_gt: (N, ) numpy array, ground truth label; 0 <= gt_label(i) <= class_num
    :param bbox_gt: numpy array, ground truth bounding box
    :param cuda: if use GPU
    :return: fast R-CNN loss
    """
    # class classification cross entropy loss
    cross_entropy_loss = nn.CrossEntropyLoss()
    gt_label = Variable(torch.from_numpy(label_gt)).type(torch.LongTensor)
    if cuda:
        gt_label.cuda()
    class_loss = cross_entropy_loss(class_pred, gt_label)

    # bounding box smooth L1 loss
    proposal_num, class_num = class_pred.size()[0:2]
    bbox_gt = Variable(torch.from_numpy(bbox_gt)).type(torch.FloatTensor)
    if cuda:
        bbox_gt.cuda()

    mask_not_background = np.repeat(np.expand_dims(label_gt > 0, axis=1), 4, axis=1).astype(np.uint8)
    mask_not_background = Variable(torch.from_numpy(mask_not_background)).type(torch.FloatTensor)
    if cuda:
        mask_not_background.cuda()

    ind_bbox_at_gt = np.zeros((proposal_num, 4 * class_num)).astype(np.uint8)
    for i in range(proposal_num):
        ind_bbox_at_gt[i, label_gt[i] * 4:(label_gt[i] + 1) * 4] = 1
    ind_bbox_at_gt = Variable(torch.from_numpy(ind_bbox_at_gt)).type(torch.ByteTensor)
    if cuda:
        ind_bbox_at_gt.cuda()
    bbox_pred_at_gt = bbox_pred[ind_bbox_at_gt].view(proposal_num, 4)

    # calculate loss
    abs_diff_pred_gt = torch.abs(bbox_pred_at_gt - bbox_gt)
    if not cuda:
        loss_larger_than_one = torch.mul((abs_diff_pred_gt >= 1).type(torch.FloatTensor),
                                         abs_diff_pred_gt - 0.5)
        loss_smaller_than_one = torch.mul((abs_diff_pred_gt < 1).type(torch.FloatTensor),
                                          torch.mul(torch.pow(abs_diff_pred_gt, 2), 0.5))
    if cuda:
        loss_larger_than_one = torch.mul((abs_diff_pred_gt >= 1).type(torch.FloatTensor).cuda(),
                                         abs_diff_pred_gt - 0.5)
        loss_smaller_than_one = torch.mul((abs_diff_pred_gt < 1).type(torch.FloatTensor).cuda(),
                                          torch.mul(torch.pow(abs_diff_pred_gt, 2), 0.5))

    bbox_loss = torch.add(loss_larger_than_one, loss_smaller_than_one)
    bbox_lost = torch.sum(torch.mul(mask_not_background, bbox_loss))

    return torch.add(class_loss, bbox_lost)


if __name__ == "__main__":
    test_proposals = np.array([[0, 0, 500, 500], [300, 300, 600, 600]])

    import cv2
    from rescale_image import rescale_image

    img_box_dict = np.load('../../VOCdevkit/img_box_dict.npy')[()]
    for img_dir, img_info in img_box_dict.items():
        image, image_info = rescale_image('../'+img_dir, img_info)
        label, gt_box_parameterized = generate_gt(image_info, test_proposals)
        print(label, gt_box_parameterized)
        for object in image_info['objects']:
            ymin, xmin, ymax, xmax = [int(i) for i in object[1:5]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(image,
                    object[0],
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))
        for box in test_proposals:
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        class_pred = Variable(torch.rand(test_proposals.shape[0], 21), requires_grad=True)
        print('class_pred.size(): ', class_pred.size())
        bbox_pred = Variable(torch.rand(test_proposals.shape[0], 84), requires_grad=True)
        loss = generate_loss(class_pred, bbox_pred, label, gt_box_parameterized)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()