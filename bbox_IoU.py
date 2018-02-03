import numpy as np


def bbox_IoU(boxes, query_boxes):
    """
    :param boxes: (N, 4) ndarray
    :param query_boxes: (K, 4) ndarray
    :return: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    IoU = np.zeros((N, K), dtype=np.float)
    for k in range(K):
        box_area = np.multiply(query_boxes[k, 2] - query_boxes[k, 0] + 1,
                              query_boxes[k, 3] - query_boxes[k, 1] + 1)
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
                    union_area = (
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - intersection_area
                    )
                    IoU[n, k] = intersection_area / union_area
    return IoU


if __name__ == "__main__":
    box = np.array([[-79,  -33,   94,   48],
                    [-165,  -75,  180,   90],
                    [-338, -157,  353,  172],
                    [-52,  -52,   67,  67],
                    [-112, -112,  127,  127],
                    [-232, -232,  247,  247],
                    [-33,  -79,   48,   94],
                    [-75, -165,   90,  180],
                    [-157, -338,  172,  353]])
    IoU = bbox_IoU(box, box)
    print(np.around(IoU, decimals=2))

    box1 = np.array([[0, 0, 2, 1]])
    box2 = np.array([[0, 0, 1, 2]])
    IoU = bbox_IoU(box1, box2)
    print(np.around(IoU, decimals=2))

    box1 = np.array([[0, 0, 2, 1]])
    box2 = np.array([[10, 10, 11, 12]])
    IoU = bbox_IoU(box1, box2)
    print(np.around(IoU, decimals=2))