import numpy as np
import torch
import cv2
import torch.nn as nn

from torchvision import transforms
from torch.autograd import Variable

from generate_anchors import generate_anchors
from rescale_image import rescale_image
from box_parametrize import box_deparameterize
from non_maximum_suppresion import non_maximum_suppression

BASE_SIZE = 16
RATIOS = [0.5, 1.0, 2.0]
SCALES = [8, 16, 32]


def generate_rpn_proposals(img, rpn_model, base_size, ratios, scales, score_threshold=0.5, iou_threshold=0.5, cuda=0):
    """
    generate rpn proposals
    :param img: (H, W, 3) ndarray
    :param rpn_model: rpn_model
    :param base_size: base size of rpn stride
    :param ratios: ratios of anchors
    :param scales: scales of anchors
    :param score_threshold: objectiveness threshold
    :param iou_threshold: IoU threshold
    :param cuda: if use GPU
    :return: (N, 4) ndarray regions proposals
    """

    # normalize image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = Variable(normalize(torch.from_numpy(np.transpose(img / 255, (2, 0, 1))))).float()
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if cuda == 1:
        img_tensor = img_tensor.cuda()

    # produce prediction
    cls_score, reg_score = rpn_model(img_tensor)

    # softmax of cls_scores
    num_anchors = int(cls_score.size()[1] / 2)
    softmax = nn.Softmin(dim=1)
    for i in range(0, num_anchors):
        cls_score[:, (2 * i, 2 * i + 1), :, :] = \
            softmax(cls_score[:, (2 * i, 2 * i + 1), :, :])

    # generate anchors
    score_dim = cls_score.shape[2:4]
    anchor_list = generate_anchors(base_size=base_size, ratios=ratios, scales=scales)
    score_height, score_width = int(score_dim[0]), int(score_dim[1])

    anchor_x_shift = np.arange(0, score_width) * base_size + base_size / 2
    anchor_y_shift = np.arange(0, score_height) * base_size + base_size / 2
    anchor_centers = np.array([[i, j]
                               for i in anchor_y_shift
                               for j in anchor_x_shift])
    all_anchors = np.array([np.tile(anchor_centers[i], 2) + anchor_list[j]
                            for i in range(0, anchor_centers.shape[0])
                            for j in range(0, anchor_list.shape[0])])

    # transform to numpy array
    cls_score = cls_score.data.numpy()
    reg_score = reg_score.data.numpy()

    # reshape
    cls_score = np.reshape(np.transpose(cls_score, (0, 2, 3, 1)), (-1, 2), order='C')
    reg_score = np.reshape(np.transpose(reg_score, (0, 2, 3, 1)), (-1, 4), order='C')

    # deparameterize bbox
    box_pred = box_deparameterize(reg_score, all_anchors)

    # perform non maximum suppression
    _, _, box_selected = non_maximum_suppression(cls_score, box_pred, (0,), score_threshold, iou_threshold)

    return box_selected


def largest_indices(ary, n):
    """
    Returns the n largest indices from a numpy array
    :param ary: ndarray
    :param n: number of top largest elements
    :return: indices
    """
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


if __name__ == '__main__':
    model_dir = 'rpn_trained.pkl'
    img_dict_dir = '../VOCdevkit/img_box_dict.npy'
    img_dict = np.load(img_dict_dir)[()]
    cuda = torch.cuda.is_available()
    if not cuda:
        net = torch.load(model_dir, map_location=lambda storage, loc: storage)
    if cuda:
        net = torch.load(model_dir)
        net.cuda()

    for img_dir, img_info in img_dict.items():
        img, modified_img_info = rescale_image(img_dir, img_info)
        box_selected = generate_rpn_proposals(
            img,
            net,
            BASE_SIZE,
            RATIOS,
            SCALES,
            score_threshold=0.5,
            iou_threshold=0.5,
            cuda=cuda)
        for object in img_info['objects']:
            ymin, xmin, ymax, xmax = [int(i) for i in object[1:5]]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(img,
                        object[0],
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))

        # draw positive predictions
        for box in box_selected[0:10]:
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            color = np.squeeze([np.random.randint(255, size=1),
                                np.random.randint(255, size=1),
                                np.random.randint(255, size=1)]).tolist()
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()