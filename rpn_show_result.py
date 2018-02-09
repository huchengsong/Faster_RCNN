import numpy as np
import torch
import cv2
import torch.nn as nn

from torchvision import transforms
from torch.autograd import Variable

from generate_anchors import generate_anchors
from rescale_image import rescale_image
from box_parametrize import box_deparameterize

BASE_SIZE = 16
RATIOS = [0.5, 1.0, 2.0]
SCALES = [8, 16, 32]


def show_rpn_result(img_dict_dir, model_dir, top_num, base_size, ratios, scales, cuda):
    """"""

    img_dict = np.load(img_dict_dir)[()]
    if not cuda:
        net = torch.load(model_dir, map_location=lambda storage, loc: storage)
    if cuda:
        net = torch.load(model_dir)
        net.cuda()

    for img_dir, img_info in img_dict.items():
        img, modified_img_info = rescale_image(img_dir, img_info)
        # normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = Variable(normalize(torch.from_numpy(np.transpose(img / 255, (2, 0, 1))))).float()
        img_tensor = torch.unsqueeze(img_tensor, 0)
        if cuda == 1:
            img_tensor = img_tensor.cuda()
        cls_score, reg_score = net(img_tensor)

        # softmax of cls_scores
        num_anchors = int(cls_score.size()[1] / 2)
        softmax = nn.Softmin(dim=1)
        for i in range(0, num_anchors):
            cls_score[:, (2 * i, 2 * i + 1), :, :] = \
                softmax(cls_score[:, (2 * i, 2 * i + 1), :, :])
        # only take the positive anchors
        cls_score_positive = cls_score[:, [2 * i for i in range(num_anchors)], :, :]

        # transform to numpy array
        cls_score_positive = cls_score_positive.data.numpy()
        reg_score = reg_score.data.numpy()

        # generate anchors
        score_dim = cls_score.size()[2:4]
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
        all_anchors = np.reshape(
            all_anchors,
            [1, score_dim[0], score_dim[1], 4 * anchor_list.shape[0]],
            order='C')
        all_anchors = np.transpose(all_anchors, (0, 3, 1, 2))

        def largest_indices(ary, n):
            """Returns the n largest indices from a numpy array."""
            flat = ary.flatten()
            indices = np.argpartition(flat, -n)[-n:]
            indices = indices[np.argsort(-flat[indices])]
            return np.unravel_index(indices, ary.shape)

        # indices of the largest n score
        indices = largest_indices(cls_score_positive, top_num)
        reg_top = np.column_stack((reg_score[indices[0], 4 * indices[1], indices[2], indices[3]],
                                   reg_score[indices[0], 4 * indices[1] + 1, indices[2], indices[3]],
                                   reg_score[indices[0], 4 * indices[1] + 2, indices[2], indices[3]],
                                   reg_score[indices[0], 4 * indices[1] + 3, indices[2], indices[3]]))
        anchor_top = np.column_stack((all_anchors[indices[0], 4 * indices[1], indices[2], indices[3]],
                                      all_anchors[indices[0], 4 * indices[1] + 1, indices[2], indices[3]],
                                      all_anchors[indices[0], 4 * indices[1] + 2, indices[2], indices[3]],
                                      all_anchors[indices[0], 4 * indices[1] + 3, indices[2], indices[3]]))
        print(reg_top)
        print(anchor_top)
        pred_boxex = box_deparameterize(reg_top, anchor_top)
        print(pred_boxex)

        # draw ground truth box on image
        for object in img_info['objects']:
            ymin, xmin, ymax, xmax = [int(i) for i in object[1:5]]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(img,
                        object[0],
                        (xmin, ymin),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))

        # draw positive predictions
        for box in pred_boxex:
            ymin, xmin, ymax, xmax = [int(i) for i in box]
            color = np.squeeze([np.random.randint(255, size=1),
                                np.random.randint(255, size=1),
                                np.random.randint(255, size=1)]).tolist()
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

        # for i in range(score_dim[0]):
        #     for j in range(score_dim[1]):
        #         for k in range(len(ratios)*len(scales)):
        #             anchor = all_anchors[0, k*4:(k+1)*4, i, j]
        #             ymin, xmin, ymax, xmax = [int(i) for i in anchor]
        #             color = np.squeeze([np.random.randint(255, size=1),
        #                                     np.random.randint(255, size=1),
        #                                     np.random.randint(255, size=1)]).tolist()
        #             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    model_dir = 'rpn_trained.pkl'
    img_dict_dir = '../VOCdevkit/img_box_dict.npy'
    top_num = 20
    show_rpn_result(img_dict_dir,
                    model_dir,
                    top_num,
                    BASE_SIZE,
                    RATIOS,
                    SCALES,
                    cuda=torch.cuda.is_available())