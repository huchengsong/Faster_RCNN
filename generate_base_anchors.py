import numpy as np
import torch

from configure import Config


def generate_base_anchors(base_size=Config.stride, ratios=Config.ratios, scales=Config.scales):
    """
    Generate anchor with reference window [0, 0, 15, 15]
    Reference window: [up_left_x, up_left_y, bottom_right_x, bottom_right_y]
    """
    offset = (base_size-1)/2
    base_anchor = np.array([-offset, -offset, offset, offset], dtype=np.float32)
    anchors = ratio_scale_enum(base_anchor, ratios, scales)
    return anchors


def get_whctr(anchor):
    """
    Return w, h, x_center, and y_center for an anchor.
    """
    h = anchor[2] - anchor[0] + 1
    w = anchor[3] - anchor[1] + 1
    y_ctr = 0.5 * (anchor[0] + anchor[2])
    x_ctr = 0.5 * (anchor[1] + anchor[3])
    return h, w, y_ctr, x_ctr


def ratio_scale_enum(anchor, ratios, scales):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    h, w, y_ctr, x_ctr = get_whctr(anchor)
    size = w * h
    ws = np.sqrt(size / ratios)
    hs = ws * ratios
    anchors = np.transpose([[y_ctr - 0.5 * scales[i] * (hs - 1) for i in range(len(scales))],
                            [x_ctr - 0.5 * scales[i] * (ws - 1) for i in range(len(scales))],
                            [y_ctr + 0.5 * scales[i] * (hs - 1) for i in range(len(scales))],
                            [x_ctr + 0.5 * scales[i] * (ws - 1) for i in range(len(scales))]]).astype(np.float32)
    return anchors.reshape(len(ratios)*len(scales), 4)


def anchor_proposals(feature_height, feature_width, stride, anchor_base):
    """
    return anchor proposals for an image
    :param feature_height: height of feature map
    :param feature_width: width of feature map
    :param stride: stride on images
    :param anchor_base: [K, 4], ndarray, anchors base at each location
    :return: [N, 4], pytroch tensor, anchor proposals
    """
    anchor_base = torch.from_numpy(anchor_base).float().cuda()
    num_base = anchor_base.size()[0]
    anchor_x_shift = torch.arange(0, feature_width) * stride + stride / 2
    anchor_x_shift = anchor_x_shift.float().cuda()
    anchor_y_shift = torch.arange(0, feature_height) * stride + stride / 2
    anchor_y_shift = anchor_y_shift.float().cuda()

    anchor_x_shift = anchor_x_shift.expand(feature_height, num_base, -1).permute(0, 2, 1)
    anchor_y_shift = anchor_y_shift.expand(feature_width, num_base, -1).permute(2, 0, 1)
    anchor_centers = torch.stack((anchor_y_shift, anchor_x_shift,
                                  anchor_y_shift, anchor_x_shift), dim=3)
    anchor_base = anchor_base.expand(feature_height, feature_width, -1, -1)
    anchors = anchor_centers + anchor_base
    return anchors


def test():
    a = generate_base_anchors(ratios=[0.5, 1, 2])
    print(a)

    # test anchor_proposals()
    anchor_base = generate_base_anchors(16, [0.5, 1.0, 2.0], [8, 16, 32])
    anchors = anchor_proposals(16, 16, 16, anchor_base).view(-1, 4)

    center = (anchors[:, [0, 1]] + anchors[:, [2, 3]]) / 2
    print(anchors[:100])
    print(center[[9 * i for i in range(16 * 16)], :] / 16)


if __name__ == '__main__':
    test()