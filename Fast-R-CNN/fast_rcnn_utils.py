import numpy as np

from bbox_IoU import bbox_IoU
from box_parametrize import box_parameterize

KEY = ['background', 'aeroplane', 'bicycle', 'bird',
       'boat', 'bottle', 'bus', 'car',
       'cat', 'chair', 'cow', 'diningtable',
       'dog', 'horse', 'motorbike', 'person',
       'pottedplant', 'sheep', 'sofa', 'train',
       'tvmonitor']


def generate_gt(img_info, region_proposals):
    """
    :param img_info: dictionary with key ('img_size', 'objects')
    :param region_proposals: (N, 4) ndarray, region proposal
    :return: label: array with N labels
    :return: gt_box_parameterized: (N, 4) ndarray, parametreized ground truth box based on
     region_proposals
    """
    label = np.zeros(region_proposals.shape[0]).astype(np.int)
    ground_truth_boxes = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
    ground_truth_class = np.array([KEY.index(cls) for cls in np.array(img_info['objects'])[:, 0]])

    # label each region proposals
    iou_matrix = bbox_IoU(ground_truth_boxes, region_proposals)
    print(iou_matrix, '\n\r')
    ind_max_each_proposal = np.argmax(iou_matrix, axis=0)
    print(ind_max_each_proposal, '\n\r')
    max_iou_each_proposal = iou_matrix[ind_max_each_proposal, np.arange(region_proposals.shape[0])]
    ind_positive = np.where(max_iou_each_proposal >= 0.5)
    label[ind_positive] = ground_truth_class[ind_max_each_proposal[ind_positive]]
    ind_negative = np.where(np.logical_and(
        max_iou_each_proposal < 0.5,
        max_iou_each_proposal >= 0.1))
    label[ind_negative] = 0

    # proposal parameterize based on ground truth
    gt_box_parameterized = box_parameterize(ground_truth_boxes[ind_max_each_proposal], region_proposals)
    gt_box_parameterized[np.where(label == 0), :] = 0

    return label, gt_box_parameterized


if __name__ == "__main__":
    test_proposals = np.array([[0, 0, 500, 500], [300, 300, 600, 600]])

    import cv2
    from rescale_image import rescale_image

    img_box_dict = np.load('../../VOCdevkit/img_box_dict.npy')[()]
    for img_dir, img_info in img_box_dict.items():
        image, modified_image_info = rescale_image('../'+img_dir, img_info)
        label, gt_box_parameterized = generate_gt(modified_image_info, test_proposals)
        print(label, gt_box_parameterized)
        for object in modified_image_info['objects']:
            ymin, xmin, ymax, xmax = [int(i) for i in object[1:5]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(image,
                    object[0],
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))
        for box in test_proposals:
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()