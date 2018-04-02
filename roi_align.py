import torch
from torch.autograd import Variable
import torch.nn as nn


class ROIAlign(nn.Module):
    def __init__(self, pool_out_size, sub_sample=2):
        super(ROIAlign, self).__init__()
        self.pool_out_size = pool_out_size
        self.sub_sample = sub_sample

    def forward(self, features, rois, ratio):
        """
        bilinear interpolation
        :param features: pytorch Variable (1, C, H, W)
        :param rois: pytorch tensor, (N, 4)
        :param ratio: ratio of feature size to image size
        :return: (N, C, H_out, W_out)
        """
        feature_size = list(features.size())
        rois_in_features = rois * ratio
        h_step = ((rois_in_features[:, 2] - rois_in_features[:, 0]) / (self.pool_out_size[0] * self.sub_sample))[:, None]
        w_step = ((rois_in_features[:, 3] - rois_in_features[:, 1]) / (self.pool_out_size[1] * self.sub_sample))[:, None]
        y_shift = torch.arange(0, self.pool_out_size[0] * self.sub_sample).cuda().expand(rois.size(0), -1) * h_step + \
                h_step / 2 + rois_in_features[:, 0][:, None]
        x_shift = torch.arange(0, self.pool_out_size[1] * self.sub_sample).cuda().expand(rois.size(0), -1) * w_step + \
                w_step / 2 + rois_in_features[:, 1][:, None]
        y_shift = y_shift.expand(self.pool_out_size[1] * self.sub_sample, -1, -1).permute(1, 2, 0)
        x_shift = x_shift.expand(self.pool_out_size[0] * self.sub_sample, -1, -1).permute(1, 0, 2)

        centers = torch.stack((y_shift, x_shift), dim=3)
        centers = centers.contiguous().view(-1, 2)  # (N, H, W, 2) -> (N*H*W, 2)

        # bilinear interpolation
        loc_y = Variable(torch.frac(centers[:, 0].expand(feature_size[0], feature_size[1], -1)))
        loc_x = Variable(torch.frac(centers[:, 1].expand(feature_size[0], feature_size[1], -1)))

        ind_left = torch.floor(centers[:, 1]).long().clamp(0, feature_size[3] - 1)
        ind_right = torch.ceil(centers[:, 1]).long().clamp(0, feature_size[3] - 1)
        ind_up = torch.floor(centers[:, 0]).long().clamp(0, feature_size[2] - 1)
        ind_down = torch.ceil(centers[:, 0]).long().clamp(0, feature_size[2] - 1)

        pre_pool = features[:, :, ind_up, ind_left] * (1 - loc_y) * (1 - loc_x) + \
                   features[:, :, ind_down, ind_left] * loc_y * (1 - loc_x) + \
                   features[:, :, ind_up, ind_right] * (1 - loc_y) * loc_x + \
                   features[:, :, ind_down, ind_right] * loc_y * loc_x

        pre_pool = pre_pool.view(feature_size[0] * feature_size[1], rois.size()[0],
                                 self.pool_out_size[0] * self.sub_sample,
                                 self.pool_out_size[1] * self.sub_sample).permute(1, 0, 2, 3)
        max_pool = nn.MaxPool2d(kernel_size=self.sub_sample, stride=self.sub_sample, padding=0)
        post_pool = max_pool(pre_pool)

        return post_pool
