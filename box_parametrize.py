import numpy as np


def box_parameterize(input_boxes, anchor_boxes):
    """
    :param input_boxes: (N, 4)
    :param anchor_boxes: (N, 4)
    :return: parameterized boxes based on anchor_boxes
    """
    input_boxes_height = input_boxes[:, 2] - input_boxes[:, 0] + 1
    input_boxes_width = input_boxes[:, 3] - input_boxes[:, 1] + 1
    input_boxes_ctr_y = (input_boxes[:, 2] + input_boxes[:, 0]) / 2
    input_boxes_ctr_x = (input_boxes[:, 3] + input_boxes[:, 1]) / 2

    anchor_boxes_height = anchor_boxes[:, 2] - anchor_boxes[:, 0] + 1
    anchor_boxes_width = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1
    anchor_boxes_ctr_y = (anchor_boxes[:, 2] + anchor_boxes[:, 0]) / 2
    anchor_boxes_ctr_x = (anchor_boxes[:, 3] + anchor_boxes[:, 1]) / 2

    result_ctr_x = (input_boxes_ctr_x - anchor_boxes_ctr_x) / anchor_boxes_width
    result_ctr_y = (input_boxes_ctr_y - anchor_boxes_ctr_y) / anchor_boxes_height
    result_width = np.log(input_boxes_width / anchor_boxes_width)  # natural logarithm
    result_height = np.log(input_boxes_height / anchor_boxes_height)  # natural logarithm

    result = np.vstack(
        (result_ctr_y, result_ctr_x, result_height, result_width)
    ).transpose()

    return result

def box_deparameterize(input_boxes, anchor_boxes):
    """
    :param input_boxes: (N, 4)
    :param anchor_boxes: (N, 4)
    :return: de-parameterized boxes based on anchor_boxes

    """
    anchor_boxes_height = anchor_boxes[:, 2] - anchor_boxes[:, 0] + 1
    anchor_boxes_width = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1
    anchor_boxes_ctr_y = (anchor_boxes[:, 2] + anchor_boxes[:, 0]) / 2
    anchor_boxes_ctr_x = (anchor_boxes[:, 3] + anchor_boxes[:, 1]) / 2

    result_height = np.exp(input_boxes[:, 2]) * anchor_boxes_height
    result_width = np.exp(input_boxes[:, 3]) * anchor_boxes_width
    result_ctr_y = input_boxes[:, 0] * anchor_boxes_height + anchor_boxes_ctr_y
    result_ctr_x = input_boxes[:, 1] * anchor_boxes_width + anchor_boxes_ctr_x

    input_boxes_ymin = result_ctr_y - (result_height - 1) / 2
    input_boxes_xmin = result_ctr_x - (result_width - 1) / 2
    input_boxes_ymax = result_ctr_y + (result_height - 1) / 2
    input_boxes_xmax = result_ctr_x + (result_width - 1) / 2

    result = np.vstack(
        (input_boxes_ymin, input_boxes_xmin, input_boxes_ymax, input_boxes_xmax)
    ).transpose()

    return result


if __name__ == '__main__':
    in_box = np.array([[0, 0, 100, 100], [-50, -50, 150, 150]])
    anchor_box = np.array([[25, 25, 75, 75], [0, 0, 100, 50]])
    out = box_parameterize(in_box, anchor_box)
    de_out = box_deparameterize(out, anchor_box)
    print(out)
    print(de_out)