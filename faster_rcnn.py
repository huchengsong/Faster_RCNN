import torch
from non_maximum_suppression import non_maximum_suppression_roi
from torch import nn

from configure import Config
from box_parametrize import box_deparameterize_gpu


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, num_class=Config.num_class):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.num_class = num_class
        self.optimizer = None

    def forward(self, x):
        """
        return predictions
        :param x: pytorch Variable, extracted features
        :return: pytorch Variable: roi_cls_locs (N, 4 * num_classes),
                                   roi_scores (N, number_classes),
                                   rois (N, 4)
        """
        img_size = x.size()[2:4]
        x = self.extractor(x)
        _, _, rois, _ = self.rpn(x, img_size)
        roi_cls_locs, roi_scores = self.head(x, rois)

        return roi_cls_locs, roi_scores, rois

    def predict(self, img_tensor):
        """
        bounding box prediction
        :param img_tensor: preprocessed image tensor
        :param iou_thresh: iou threshold for nms
        :param score_thresh: score threshold
        :return: ndarray: label (N, ), score (N, ), box (N, 4)
        """
        # self.eval() set the module in evaluation mode: self.train(False)
        self.eval()

        # store training parameters
        train_score_thresh = Config.score_thresh
        train_iou_thresh = Config.iou_thresh
        train_num_pre_nms = Config.num_pre_nms
        train_num_post_nms = Config.num_post_nms

        # set parameters for evaluation
        Config.score_thresh = 0.05
        Config.iou_thresh = 0.3
        Config.num_pre_nms = 6000
        Config.num_post_nms = 300

        img_size = img_tensor.size()[2:4]

        roi_cls_loc, roi_scores, rois = self(img_tensor)
        roi_scores = nn.Softmax(dim=1)(roi_scores).data
        roi_cls_loc = roi_cls_loc.view(-1, 4).data

        rois = rois.view(-1, 1, 4).repeat(1, self.num_class, 1).view(-1, 4)
        cls_bbox = box_deparameterize_gpu(roi_cls_loc, rois)

        # clip bounding boxes
        cls_bbox[:, [0, 2]] = cls_bbox[:, [0, 2]].clamp(0, img_size[0])
        cls_bbox[:, [1, 3]] = cls_bbox[:, [1, 3]].clamp(0, img_size[1])
        cls_bbox = cls_bbox.view(-1, self.num_class * 4)

        box, score, label = non_maximum_suppression_roi(roi_scores, cls_bbox, range(1, Config.num_class),
                                                        score_thresh=Config.score_thresh, iou_thresh=Config.iou_thresh)
        self.train()

        # restore parameter for training
        Config.score_thresh = train_score_thresh
        Config.iou_thresh = train_iou_thresh
        Config.num_pre_nms = train_num_pre_nms
        Config.num_post_nms = train_num_post_nms

        return box, score, label

    def get_optimizer(self, lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params.append({'params': [value], 'lr': lr * 2, 'weight_decay': 0})
                else:
                    params.append({'params': [value], 'lr': lr, 'weight_decay': Config.weight_decay})
        self.optimizer = torch.optim.SGD(params, momentum=0.9)

    def scale_lr(self, decay):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
