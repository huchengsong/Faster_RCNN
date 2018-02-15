import numpy as np
import cv2
import torch
import torch.nn as nn

from torchvision import transforms
from torch.autograd import Variable

from rescale_image import rescale_image
from generate_rpn_proposals import generate_rpn_proposals
from box_parametrize import box_deparameterize
from non_maximum_suppresion import non_maximum_suppression

BASE_SIZE = 16
RATIOS = [0.5, 1.0, 2.0]
SCALES = [8, 16, 32]
KEY = ['background', 'aeroplane', 'bicycle', 'bird',
       'boat', 'bottle', 'bus', 'car',
       'cat', 'chair', 'cow', 'diningtable',
       'dog', 'horse', 'motorbike', 'person',
       'pottedplant', 'sheep', 'sofa', 'train',
       'tvmonitor']


def show_fast_rcnn_result(rpn_model_dir, fast_rcnn_model_dir, img_dict_dir, cuda):

    img_dict = np.load(img_dict_dir)[()]

    if not cuda:
        fast_rcnn_net = torch.load(fast_rcnn_model_dir, map_location=lambda storage, loc: storage)
    if cuda:
        fast_rcnn_net = torch.load(fast_rcnn_model_dir)
        fast_rcnn_net.cuda()

    if not cuda:
        rpn_net = torch.load(rpn_model_dir, map_location=lambda storage, loc: storage)
    if cuda:
        rpn_net = torch.load(rpn_model_dir)
        rpn_net.cuda()

    for img_dir, img_info in img_dict.items():
        img, img_info = rescale_image(img_dir, img_info)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = Variable(normalize(torch.from_numpy(np.transpose(img / 255, (2, 0, 1))))).float()
        img_tensor = torch.unsqueeze(img_tensor, 0)
        if cuda == 1:
            img_tensor = img_tensor.cuda()

        region_proposals = generate_rpn_proposals(
            img_tensor,
            rpn_net,
            BASE_SIZE,
            RATIOS,
            SCALES,
            score_threshold=0.3,
            iou_threshold=0.7,
            cuda=cuda)

        if region_proposals.shape[0] > 128:
            region_proposals = region_proposals[0:128, :]

        cls_pred, bbox_pred = fast_rcnn_net(img_tensor, img_info, region_proposals)

        # process predictions
        cls_pred = cls_pred.data.cpu().numpy()
        bbox_pred = bbox_pred.data.cpu().numpy()

        cls_argmax = np.argmax(cls_pred, axis=1)
        boxes_argmax = []
        for i in range(len(cls_argmax)):
            boxes_argmax.append(bbox_pred[i, 4 * cls_argmax[i]:4 * (cls_argmax[i] + 1)])
        boxes_argmax = np.array(boxes_argmax)

        boxes_argmax = box_deparameterize(boxes_argmax, region_proposals)
        class_label, box_score, boxes = non_maximum_suppression(cls_pred, boxes_argmax, np.arange(1, 21), 0.0, 0.5, ignore_argmax_pred=False)

        # draw predictions
        for i in range(len(class_label)):
            box = boxes[i, :]
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            cv2.putText(img,
                        KEY[class_label[i]] + ' ' + str(box_score[i]),
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))
            color = np.squeeze([np.random.randint(255, size=1),
                                np.random.randint(255, size=1),
                                np.random.randint(255, size=1)]).tolist()
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rpn_model_dir = 'rpn_trained.pkl'
    fast_rcnn_model_dir = 'fast_rcnn_trained.pkl'
    img_dict_dir = '../VOCdevkit/img_box_dict.npy'
    show_fast_rcnn_result(rpn_model_dir, fast_rcnn_model_dir, img_dict_dir, torch.cuda.is_available())
