import numpy as np
from torch import nn
import torch
from rpn_utils import generate_anchor_loc_label, generate_training_anchors
from torch.autograd import Variable

from convert_label import text_to_num
from rpn_utils import rpn_loss
from fast_rcnn_utils import fast_rcnn_loss


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, rpn_sigma=3, roi_sigma=1):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.optimizer = self.faster_rcnn.optimizer

    def forward(self, img_tensor, img_info, img):
        img_size = img_info['img_size']
        features = self.faster_rcnn.extractor(img_tensor)

        gt_bbox = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
        gt_label = np.array(img_info['objects'])[:, 0]
        gt_label = text_to_num(gt_label)

        # RPN loss
        rpn_locs, rpn_scores, rois, anchors = self.faster_rcnn.rpn(features, img_size)
        gt_rpn_label, gt_rpn_loc = generate_anchor_loc_label(anchors, gt_bbox, img_size)
        rpn_cls_loss, rpn_loc_loss = rpn_loss(rpn_scores, rpn_locs,
                                              gt_rpn_loc, gt_rpn_label,
                                              self.rpn_sigma)
        print('rpn_cls_loss', rpn_cls_loss.data.cpu().numpy(), 'rpn_loc_loss', rpn_loc_loss.data.cpu().numpy())

        # generate proposals from rpn rois
        sampled_roi, gt_roi_loc, gt_roi_label = generate_training_anchors(rois, gt_bbox, gt_label)

        # ROI loss
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sampled_roi)
        roi_cls_loss, roi_loc_loss = fast_rcnn_loss(roi_score, roi_cls_loc,
                                                    gt_roi_loc, gt_roi_label,
                                                    self.roi_sigma)
        print('roi_cls_loss', roi_cls_loss.data.cpu().numpy(), 'roi_loc_loss', roi_loc_loss.data.cpu().numpy())

        # ########################################
        # ########### test code ##################
        # ########################################
        """
        import cv2
        import torch
        from box_parametrize import box_deparameterize_gpu
        # show gt anchors
        anchor_cpu = anchors[torch.nonzero(gt_rpn_label == 1).squeeze_()].cpu().numpy()
        print(anchor_cpu.shape)
        for box in anchor_cpu:
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # show deparameterized gt_rpn_loc
        bboxes = box_deparameterize_gpu(gt_rpn_loc, anchors)
        bboxes = bboxes[torch.nonzero(gt_rpn_label == 1).squeeze_()].cpu().numpy()
        print(bboxes.shape)
        for box in bboxes:
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # show gt rois
        KEY = ['background', 'aeroplane', 'bicycle', 'bird',
               'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

        rois_gt = sampled_roi[torch.nonzero(gt_roi_label>0).squeeze_()].cpu().numpy()
        rois_label = gt_roi_label[torch.nonzero(gt_roi_label>0).squeeze_()].cpu().numpy()
        print(rois_gt.shape)
        for i in range(rois_gt.shape[0]):
            box = rois_gt[i]
            ymin, xmin, ymax, xmax = [int(j) for j in box]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(img,
                        KEY[rois_label[i]],
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # test ground truth roi loc
        roi_loc_back = box_deparameterize_gpu(gt_roi_loc, sampled_roi)
        roi_loc_back = roi_loc_back[torch.nonzero(gt_roi_label>0).squeeze_()].cpu().numpy()
        for i in range(roi_loc_back.shape[0]):
            box = roi_loc_back[i]
            ymin, xmin, ymax, xmax = [int(j) for j in box]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        # ########################################
        # ########### test code ##################
        # ########################################

        return rpn_cls_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss

    def train_step(self, img_tensor, img_info, img):
        self.optimizer.zero_grad()
        loss = self.forward(img_tensor, img_info, img)
        print('total loss', loss.data.cpu().numpy())
        loss.backward()
        self.optimizer.step()
