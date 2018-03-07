import torch
from non_maximum_suppression import non_maximum_suppression_roi
from torch.autograd import Variable
import numpy as np
from torch import nn

from box_parametrize import box_deparameterize_gpu


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=[0., 0., 0., 0.],
                 loc_normalize_std=[0.1, 0.1, 0.2, 0.2],
                 num_class=21):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.num_class = num_class

        self.optimizer = self.get_optimizer(lr=1e-3)

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
        _, _, _, rois = self.rpn(x, img_size)
        roi_cls_locs, roi_scores = self.head(x, rois)

        return roi_cls_locs, roi_scores, rois

    def predict(self, img_tensor, nms_thresh=0.3, score_thresh=0.7):
        """
        bounding box prediction
        :param img_tensor: preprocessed image tensor
        :param nms_thresh: nms threshold
        :param score_thresh: score threshold
        :return: ndarray: label (N, ), score (N, ), box (N, 4)
        """
        # self.eval() set the module in training mode: self.train(False)
        self.eval()
        img_size = img_tensor.size()[2:4]

        img_tensor = Variable(img_tensor, volatile=True)
        roi_cls_loc, roi_scores, rois = self(img_tensor)
        roi_scores = nn.Softmax(dim=1)(roi_scores.data).data

        # (1, 4*n_class)
        mean = torch.cuda.FloatTensor(self.loc_normalize_mean).expand(self.num_class, -1).contiguous().view(1, -1)
        std = torch.cuda.FloatTensor(self.loc_normalize_std).expand(self.num_class, -1).contiguous().view(1, -1)

        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, 4)

        rois = rois.view(-1, 1, 4).repeat(1, self.num_class, 1).view(-1, 4)
        cls_bbox = box_deparameterize_gpu(roi_cls_loc, rois)

        # clip bounding boxes
        cls_bbox[: [0, 2]] = cls_bbox[: [0, 2]].clamp(0, img_size[0])
        cls_bbox[: [1, 3]] = cls_bbox[: [1, 3]].clamp(0, img_size[1])
        cls_bbox = cls_bbox.view(-1, self.num_class * 4)

        label, score, box = non_maximum_suppression_roi(roi_scores, cls_bbox,
                                                        np.arange(1, 21),
                                                        score_thresh, nms_thresh)
        self.train()
        return label, score, box

    def get_optimizer(self, lr=1e-3):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params.append({'params': [value], 'lr': lr * 2, 'weight_decay': 0})
                else:
                    params.append({'params': [value], 'lr': lr, 'weight_decay': 0.0005})
        optimizer = torch.optim.Adam(params)
        return optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
