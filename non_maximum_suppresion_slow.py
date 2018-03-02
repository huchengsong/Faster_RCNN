import numpy as np


def non_maximum_suppression(box_scores, bboxes, class_list, score_threshold, iou_threshold):
    """
    using non maximum suppression to reduce bbox number
    :param box_scores: (N, class_num) ndarray
    :param bboxes: (N, 4 * class_num) ndarray
    :param class_list: list of class ID that NMS apply to
    :param score_threshold: score threshold for box selection
    :param iou_threshold: iou threshold
    :return: label, score, box
    """

    box_selected = []
    box_score = []
    class_label = []
    class_pred = np.argmax(box_scores, axis=1)
    for class_id in class_list:
        index = np.where(class_pred == class_id)
        if np.array(index).size == 0:
            continue
        score_candidate = np.squeeze(box_scores[index, class_id])
        box_candidate = np.squeeze(bboxes[index, class_id * 4:(class_id + 1) * 4])

        # delete box with score less than threshold
        ind_above_threshold = score_candidate > score_threshold
        score_candidate_above_threshold = score_candidate[ind_above_threshold]
        box_candidate_above_threshold = box_candidate[ind_above_threshold]
        box_candidate_above_threshold = np.squeeze(box_candidate_above_threshold).reshape(-1, 4)
        if np.array(score_candidate_above_threshold).size == 0:
            continue

        y1 = box_candidate_above_threshold[:, 0]
        x1 = box_candidate_above_threshold[:, 1]
        y2 = box_candidate_above_threshold[:, 2]
        x2 = box_candidate_above_threshold[:, 3]

        # index from smallest element to largest element
        idx_sorted = np.argsort(score_candidate_above_threshold)
        while len(idx_sorted) > 0:
            ind_largest_score = idx_sorted[-1]
            area_largest_score = (x2[ind_largest_score] - x1[ind_largest_score] + 1) * \
                                 (y2[ind_largest_score] - y1[ind_largest_score] + 1)
            suppression = [len(idx_sorted) - 1]
            for i in range(0, len(idx_sorted) - 1):
                index = idx_sorted[i]
                inter_w = min(x2[ind_largest_score], x2[index]) - max(x1[ind_largest_score], x1[index]) + 1
                inter_h = min(y2[ind_largest_score], y2[index]) - max(y1[ind_largest_score], y1[index]) + 1
                if (inter_w > 0) and (inter_h > 0):
                    intersection_area = inter_w * inter_h
                    union_area = (area_largest_score +
                                  (x2[index] - x1[index] + 1) * (y2[index] - y1[index] + 1) -
                                  intersection_area)
                    iou = intersection_area / union_area
                else:
                    iou = 0
                if iou > iou_threshold:
                    suppression.append(i)
            idx_sorted = np.delete(idx_sorted, suppression)
            box_selected.append(box_candidate_above_threshold[ind_largest_score])
            box_score.append(score_candidate_above_threshold[ind_largest_score])
            class_label.append(class_id)

    return np.array(class_label), np.array(box_score), np.array(box_selected)


if __name__ == '__main__':
    # def softmax(x, axis=None):
    #     return np.exp(x) / np.exp(x).sum(axis=axis)[:, None]
    # box_scores = softmax(np.random.rand(10, 3), axis=1)
    box_scores = np.array([[0.35629722, 0.29934438, 0.34435839],
                           [0.33190525, 0.43829933, 0.22979542],
                           [0.35513699, 0.39927991, 0.24558309],
                           [0.43057433, 0.33340436, 0.23602131],
                           [0.33935831, 0.20390023, 0.45674146],
                           [0.37249841, 0.30570373, 0.32179787],
                           [0.47038923, 0.23873957, 0.2908712],
                           [0.28352702, 0.31179673, 0.40467625],
                           [0.22726669, 0.47013369, 0.30259961],
                           [0.39092056, 0.27074105, 0.33833839]])

    bboxes = np.array([[0, 0, 100, 100],
                       [0, 0, 110, 110],
                       [0, 0, 125, 125],
                       [0, 0, 150, 150],
                       [0, 0, 100, 100],
                       [0, 0, 110, 110],
                       [0, 0, 125, 125],
                       [0, 0, 150, 150],
                       [500, 500, 600, 600],
                       [500, 500, 650, 650]])
    class_label, box_score, box = non_maximum_suppression(box_scores,
                                                          np.concatenate((bboxes, bboxes, bboxes), axis=1),
                                                          [0, 1, 2],
                                                          score_threshold=0,
                                                          iou_threshold=0.5)
    print(class_label, box_score, box)

