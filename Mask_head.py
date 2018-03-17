import torch
import torch.nn as nn
from torch.autograd import Variable


def roi_align(features, rois, img_info, mask_in_size, sub_sample=2):
    """
    bilinear interpolition
    :param features: pytorch Variable (1, C, H, W)
    :param rois: pytorch tensor, (N, 4)
    :param img_info: image information dictionary
    :param mask_in_size: size after ROI align
    :param mask_out_size: final mask size
    :return: (N, C, H_out, W_out)
    """
    feature_size = list(features.size())
    img_size = torch.cuda.FloatTensor(img_info['img_size'])
    feature_hw = torch.cuda.FloatTensor(feature_size[2:])
    rois_in_features = rois * feature_hw.repeat(2) / img_size.repeat(2)
    h_step = ((rois_in_features[:, 2] - rois_in_features[:, 0]) / (mask_in_size[0] * sub_sample))[:, None]
    w_step = ((rois_in_features[:, 3] - rois_in_features[:, 1]) / (mask_in_size[1] * sub_sample))[:, None]
    y_shift = torch.arange(0, mask_in_size[0] * sub_sample).cuda().expand(rois.size()[0], -1) * h_step + \
              h_step / 2 + rois_in_features[:, 0][:, None]
    x_shift = torch.arange(0, mask_in_size[1] * sub_sample).cuda().expand(rois.size()[0], -1) * w_step + \
              w_step / 2 + rois_in_features[:, 1][:, None]
    y_shift = y_shift.expand(mask_in_size[1] * sub_sample, -1, -1).permute(1, 2, 0)
    x_shift = x_shift.expand(mask_in_size[0] * sub_sample, -1, -1).permute(1, 0, 2)

    centers = torch.stack((y_shift, x_shift), dim=3).contiguous().view(-1, 2)  # (N, H, W, 2) -> (N*H*W, 2)

    # bilinear interpolition
    loc_y = Variable(centers[:, 0] - torch.floor(centers[:, 0])).expand(feature_size[0], feature_size[1], -1)
    loc_x = Variable(centers[:, 1] - torch.floor(centers[:, 1])).expand(feature_size[0], feature_size[1], -1)

    ind_left = torch.floor(centers[:, 1]).long()
    ind_right = torch.ceil(centers[:, 1]).long()
    ind_up = torch.floor(centers[:, 0]).long()
    ind_down = torch.ceil(centers[:, 0]).long()

    pre_pool = features[:, :, ind_up, ind_left] * (1 - loc_y) * (1 - loc_x) + \
               features[:, :, ind_down, ind_left] * loc_y * (1 - loc_x) + \
               features[:, :, ind_up, ind_right] * (1 - loc_y) * loc_x + \
               features[:, :, ind_down, ind_right] * loc_y * loc_x

    pre_pool = pre_pool.view(feature_size[1], rois.size()[0],
                             mask_in_size[0] * sub_sample, mask_in_size[1] * sub_sample).permute(1, 0, 2, 3)
    max_pool = nn.MaxPool2d(kernel_size=sub_sample, stride=sub_sample, padding=0)
    post_pool = max_pool(pre_pool)

    return post_pool


class MaskHead(nn.Module):
    def __init__(self):
        pass

    def forward(self, features, rois, img_info, mask_in_size, mask_out_size):
        pass



def test():
    import cv2
    import numpy as np
    img = cv2.imread('test.jpg')
    img = cv2.resize(img, (7, 7))
    img_tensor = Variable(torch.from_numpy(img).float().cuda()).permute(2, 0, 1)
    img_tensor.unsqueeze_(0)
    img_size = img.shape[:2]
    img_info = {'img_size': img_size}
    mask_in_size = [14, 14]

    rois = torch.cuda.FloatTensor([[0, 0, 7, 7], [2, 2, 3, 3]])
    result = roi_align(img_tensor, rois, img_info, mask_in_size, sub_sample=2)

    result1 = result.data.cpu()[0].permute(1, 2, 0).numpy().astype(np.uint8)
    result2 = result.data.cpu()[1].permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imshow('image', result1)
    print(result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()