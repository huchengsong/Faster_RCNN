import numpy as np
import torch

def box_parameterize(input_boxes, base_boxes):
    """
    :param input_boxes: (N, 4)
    :param base_boxes: (N, 4)
    :return: parameterized boxes based on base_boxes
    """
    input_boxes_height = input_boxes[:, 2] - input_boxes[:, 0] + 1
    input_boxes_width = input_boxes[:, 3] - input_boxes[:, 1] + 1
    input_boxes_ctr_y = (input_boxes[:, 2] + input_boxes[:, 0]) / 2
    input_boxes_ctr_x = (input_boxes[:, 3] + input_boxes[:, 1]) / 2

    base_boxes_height = base_boxes[:, 2] - base_boxes[:, 0] + 1
    base_boxes_width = base_boxes[:, 3] - base_boxes[:, 1] + 1
    base_boxes_ctr_y = (base_boxes[:, 2] + base_boxes[:, 0]) / 2
    base_boxes_ctr_x = (base_boxes[:, 3] + base_boxes[:, 1]) / 2

    result_ctr_x = (input_boxes_ctr_x - base_boxes_ctr_x) / base_boxes_width
    result_ctr_y = (input_boxes_ctr_y - base_boxes_ctr_y) / base_boxes_height
    result_width = np.log(input_boxes_width / base_boxes_width)  # natural logarithm
    result_height = np.log(input_boxes_height / base_boxes_height)  # natural logarithm

    result = np.vstack(
        (result_ctr_y, result_ctr_x, result_height, result_width)
    ).transpose()

    return result


def box_deparameterize(input_boxes, base_boxes):
    """
    :param input_boxes: (N, 4)
    :param base_boxes: (N, 4)
    :return: de-parameterized boxes based on base_boxes

    """
    # make input matrix 2d array
    input_boxes = input_boxes.reshape((-1, 4))
    base_boxes = base_boxes.reshape((-1, 4))

    base_boxes_height = base_boxes[:, 2] - base_boxes[:, 0] + 1
    base_boxes_width = base_boxes[:, 3] - base_boxes[:, 1] + 1
    base_boxes_ctr_y = (base_boxes[:, 2] + base_boxes[:, 0]) / 2
    base_boxes_ctr_x = (base_boxes[:, 3] + base_boxes[:, 1]) / 2

    result_height = np.exp(input_boxes[:, 2]) * base_boxes_height
    result_width = np.exp(input_boxes[:, 3]) * base_boxes_width
    result_ctr_y = input_boxes[:, 0] * base_boxes_height + base_boxes_ctr_y
    result_ctr_x = input_boxes[:, 1] * base_boxes_width + base_boxes_ctr_x

    input_boxes_ymin = result_ctr_y - (result_height - 1) / 2
    input_boxes_xmin = result_ctr_x - (result_width - 1) / 2
    input_boxes_ymax = result_ctr_y + (result_height - 1) / 2
    input_boxes_xmax = result_ctr_x + (result_width - 1) / 2

    result = np.vstack(
        (input_boxes_ymin, input_boxes_xmin, input_boxes_ymax, input_boxes_xmax)
    ).transpose()

    return result


def box_deparameterize_gpu(input_boxes, base_boxes):
    """
    :param input_boxes: (N, 4), pytroch Variable
    :param base_boxes: (N, 4), pytorch Varialbe
    :return: de-parameterized boxes based on base_boxes

    """
    # make input matrix 2d array
    input_boxes = input_boxes.view(-1, 4)
    base_boxes = base_boxes.view(-1, 4)

    base_boxes_height = base_boxes[:, 2] - base_boxes[:, 0] + 1
    base_boxes_width = base_boxes[:, 3] - base_boxes[:, 1] + 1
    base_boxes_ctr_y = (base_boxes[:, 2] + base_boxes[:, 0]) / 2
    base_boxes_ctr_x = (base_boxes[:, 3] + base_boxes[:, 1]) / 2

    result_height = torch.exp(input_boxes[:, 2]) * base_boxes_height
    result_width = torch.exp(input_boxes[:, 3]) * base_boxes_width
    result_ctr_y = input_boxes[:, 0] * base_boxes_height + base_boxes_ctr_y
    result_ctr_x = input_boxes[:, 1] * base_boxes_width + base_boxes_ctr_x

    input_boxes_ymin = result_ctr_y - (result_height - 1) / 2
    input_boxes_xmin = result_ctr_x - (result_width - 1) / 2
    input_boxes_ymax = result_ctr_y + (result_height - 1) / 2
    input_boxes_xmax = result_ctr_x + (result_width - 1) / 2

    result = torch.stack((input_boxes_ymin, input_boxes_xmin, input_boxes_ymax, input_boxes_xmax), dim=1)

    return result


def box_parameterize_gpu(input_boxes, base_boxes):
    """
    :param input_boxes: (N, 4), pytroch Variable
    :param base_boxes: (N, 4), pytroch Variable
    :return: parameterized boxes based on base_boxes
    """
    input_boxes_height = input_boxes[:, 2] - input_boxes[:, 0] + 1
    input_boxes_width = input_boxes[:, 3] - input_boxes[:, 1] + 1
    input_boxes_ctr_y = (input_boxes[:, 2] + input_boxes[:, 0]) / 2
    input_boxes_ctr_x = (input_boxes[:, 3] + input_boxes[:, 1]) / 2

    base_boxes_height = base_boxes[:, 2] - base_boxes[:, 0] + 1
    base_boxes_width = base_boxes[:, 3] - base_boxes[:, 1] + 1
    base_boxes_ctr_y = (base_boxes[:, 2] + base_boxes[:, 0]) / 2
    base_boxes_ctr_x = (base_boxes[:, 3] + base_boxes[:, 1]) / 2

    result_ctr_x = (input_boxes_ctr_x - base_boxes_ctr_x) / base_boxes_width
    result_ctr_y = (input_boxes_ctr_y - base_boxes_ctr_y) / base_boxes_height
    result_width = torch.log(input_boxes_width / base_boxes_width)  # natural logarithm
    result_height = torch.log(input_boxes_height / base_boxes_height)  # natural logarithm

    result = torch.stack((result_ctr_y, result_ctr_x, result_height, result_width), dim=1)

    return result

def test_correctness():
    in_box = np.array([[0, 0, 100, 100], [-50, -50, 150, 150]])
    anchor_box = np.array([[25, 25, 75, 75], [0, 0, 100, 50]])
    out = box_parameterize(in_box, anchor_box)
    de_out = box_deparameterize(out, anchor_box)
    print(out)
    print(de_out)

    from torch.autograd import Variable
    in_box = Variable(torch.from_numpy(in_box)).float().cuda()
    anchor_box = Variable(torch.from_numpy(anchor_box)).float().contiguous().cuda()
    out = box_parameterize_gpu(in_box, anchor_box)
    de_out = box_deparameterize_gpu(out, anchor_box)
    print(out)
    print(de_out)


def test_speed(num_box):
    from timeit import default_timer as timer
    array = np.empty((num_box, 4), np.float32)
    rand = np.random.randint(0, 500, size=(num_box, 2))
    array[:, 0:2] = rand
    array[:, 2:4] = rand + 200

    start = timer()
    result = box_parameterize(array, array)
    end = timer()
    print('CPU', end - start)

    start = timer()
    box_deparameterize(result, array)
    end = timer()
    print('CPU', end - start)

    # tes gpu
    array = torch.from_numpy(array).cuda()
    start = timer()
    result = box_parameterize_gpu(array, array)
    end = timer()
    print('GPU', end - start)

    start = timer()
    box_deparameterize_gpu(result, array)
    end = timer()
    print('GPU', end - start)


if __name__ == '__main__':
    test_correctness()
    test_speed(10000)
