import numpy as np
import torch
from numba import jit, float32


@jit([float32[:](float32[:], float32[:])])
def bbox_IoU(boxes, query_boxes):
    """
    :param boxes: (N, 4) ndarray
    :param query_boxes: (K, 4) ndarray
    :return: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    IoU = np.zeros((N, K), dtype=np.float32)
    query_box_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) *\
                     (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    for k in range(K):
        for n in range(N):
            intersection_h = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if intersection_h > 0:
                intersection_w = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if intersection_w > 0:
                    intersection_area = intersection_w * intersection_h
                    union_area = (boxes_area[n] + query_box_area[k] - intersection_area)
                    IoU[n, k] = intersection_area / union_area
    return IoU


def bbox_IoU_gpu(boxes, query_boxes):
    """
    :param boxes: (N, 4) pytorch tensor
    :param query_boxes: (K, 4) pytorch tensor
    :return: (N, K) pytroch tensor of overlap between boxes and query_boxes
    """
    N = boxes.size()[0]
    K = query_boxes.size()[0]
    query_box_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) *\
                     (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    query_box_area = query_box_area.expand(N, K)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    boxes_area = boxes_area.expand(K, N).t()

    intersection_h_1, _ = torch.min(torch.stack((boxes[:, 2].expand(K, N).t(),
                                                 query_boxes[:, 2].expand(N, K)), dim=0), dim=0)
    intersection_h_2, _ = torch.max(torch.stack((boxes[:, 0].expand(K, N).t(),
                                                 query_boxes[:, 0].expand(N, K)), dim=0), dim=0)
    intersection_h = intersection_h_1 - intersection_h_2 + 1

    intersection_w_1, _ = torch.min(torch.stack((boxes[:, 3].expand(K, N).t(),
                                                 query_boxes[:, 3].expand(N, K)), dim=0), dim=0)
    intersection_w_2, _ = torch.max(torch.stack((boxes[:, 1].expand(K, N).t(),
                                                 query_boxes[:, 1].expand(N, K)), dim=0), dim=0)
    intersection_w = intersection_w_1 - intersection_w_2 + 1

    intersection_area = intersection_w * intersection_h
    union_area = boxes_area + query_box_area - intersection_area
    IoU = intersection_area / union_area
    IoU = torch.clamp(IoU, min=0)
    return IoU


def test_correctness():
    box = np.array([[-79, -33, 94, 48],
                    [-165, -75, 180, 90],
                    [-338, -157, 353, 172],
                    [-52, -52, 67, 67],
                    [-112, -112, 127, 127],
                    [-232, -232, 247, 247],
                    [-33, -79, 48, 94],
                    [-75, -165, 90, 180],
                    [-157, -338, 172, 353]])
    IoU = bbox_IoU(box, box)
    print(np.around(IoU, decimals=4))

    box1 = np.array([[0, 0, 2, 1]])
    box2 = np.array([[0, 0, 1, 2]])
    IoU = bbox_IoU(box1, box2)
    print(np.around(IoU, decimals=2))

    box1 = np.array([[0, 0, 200, 100]])
    box2 = np.array([[0, 0, 100, 200]])
    IoU = bbox_IoU(box1, box2)
    print(np.around(IoU, decimals=2))

    box1 = np.array([[0, 0, 2, 1]])
    box2 = np.array([[10, 10, 11, 12]])
    IoU = bbox_IoU(box1, box2)
    print(np.around(IoU, decimals=2))

    # test gpu version
    box = torch.from_numpy(box).float().cuda()
    IoU = bbox_IoU_gpu(box, box)
    print(IoU)

    box1 = torch.from_numpy(np.array([[0, 0, 2, 1]])).float().cuda()
    box2 = torch.from_numpy(np.array([[0, 0, 1, 2]])).float().cuda()
    IoU = bbox_IoU_gpu(box1, box2)
    print(IoU)

    box1 = torch.from_numpy(np.array([[0, 0, 200, 100]])).float().cuda()
    box2 = torch.from_numpy(np.array([[0, 0, 100, 200]])).float().cuda()
    IoU = bbox_IoU_gpu(box1, box2)
    print(IoU)

    box1 = torch.from_numpy(np.array([[0, 0, 2, 1]])).float().cuda()
    box2 = torch.from_numpy(np.array([[10, 10, 11, 12]])).float().cuda()
    IoU = bbox_IoU_gpu(box1, box2)
    print(IoU)


def test_speed(num_box):
    from timeit import default_timer as timer
    array = np.empty((num_box, 4), np.float32)
    rand = np.random.randint(0, 500, size=(num_box, 2))
    array[:, 0:2] = rand
    array[:, 2:4] = rand + 200

    start = timer()
    bbox_IoU(array[0:100], array)
    end = timer()
    print('CUP:', end - start)

    # test gpu version
    array = torch.from_numpy(array).cuda()
    start = timer()
    bbox_IoU_gpu(array[0:100], array)
    end = timer()
    print('GPU:', end - start)


if __name__ == "__main__":
    # test correctness
    test_correctness()

    # test speed
    test_speed(10000)
