import numpy as np


def generate_base_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generate anchor with reference window [0, 0, 15, 15]
    Reference window: [up_left_x, up_left_y, bottom_right_x, bottom_right_y]
    """
    offset = (base_size-1)/2
    base_anchor = np.array([-offset, -offset, offset, offset])
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
                            [x_ctr + 0.5 * scales[i] * (ws - 1) for i in range(len(scales))]])
    return anchors.reshape(len(ratios)*len(scales), 4)


if __name__ == '__main__':
    a = generate_base_anchors(ratios=[0.5, 0.75, 1, 1.5, 2])
    print(a.shape)
    print((a[:, 2] - a[:, 0] + 1) / 2)
    print((a[:, 3] - a[:, 1] + 1) / 2)
    print(a)
