from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms

from faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from rescale_image import rescale_image
from convert_label import text_to_num
from eval_tool import eval_detection_voc
from configure import Config


def generate_train_val_test_data(img_dict_dir, p_train=0.7, p_val=0.1):
    """
    retrun training, validation, test subsample
    :param img_dict_dir: dictionary storing image directory and labeling
    :param p_train: ratio of training images
    :param p_val: ratio of validation images
    :return: dict_train, dict_val, dict_test
    """
    img_dict = np.load(img_dict_dir)[()]
    total_imgs = len(img_dict)
    num_train_imgs = int(total_imgs * p_train)
    num_val_imgs = int(total_imgs * p_val)
    img_dict_items = list(img_dict.items())
    np.random.shuffle(img_dict_items)
    dict_train, dict_val, dict_test = \
        np.split(img_dict_items, [num_train_imgs, num_train_imgs + num_val_imgs])
    dict_train, dict_val, dict_test = dict(dict_train), dict(dict_val), dict(dict_test)
    return dict_train, dict_val, dict_test


def create_img_tensor(img):
    """
    normalize img and convert to torch Variable
    :param img: [H, W, C] in range (0, 255)
    :return: [1, C, H, W], normalized img_tensor in range (0, 1)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = Variable(normalize(torch.from_numpy(np.transpose(img.astype(np.float32) / 255, (2, 0, 1)))))
    img_tensor = img_tensor.cuda()
    img_tensor = img_tensor.unsqueeze_(0)

    return img_tensor


def evaluation(eval_dict, faster_rcnn, test_num=Config.eval_num):
    bboxes, labels, scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for i, [img_dir, img_info] in tqdm(enumerate(eval_dict.items())):
        if len(img_info['objects']) == 0:
            continue
        img, img_info = rescale_image(img_dir, img_info)
        img_tensor = create_img_tensor(img)
        box, score, label = faster_rcnn.predict(img_tensor,
                                                score_thresh=Config.eval_score_thresh,
                                                iou_thresh=Config.eval_iou_thresh)

        gt_bbox = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
        gt_label = np.array(img_info['objects'])[:, 0]
        gt_label = text_to_num(gt_label)
        bboxes.append(list(box))
        labels.append(list(label))
        scores.append(list(score))
        gt_bboxes.append(list(gt_bbox))
        gt_labels.append(list(gt_label))
        if i == test_num:
            break

        # ########################################
        # ########### test code ##################
        # ########################################
        key = Config.class_key
        for i, b in enumerate(box):
            ymin, xmin, ymax, xmax = [int(j) for j in b]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(img,
                        key[label[i]],
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))
        for i, b in enumerate(gt_bbox):
            ymin, xmin, ymax, xmax = [int(j) for j in b]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(img,
                        key[gt_label[i]],
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0))
        cv2.imshow('image', img[:, :, [2, 1, 0]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ########################################
        # ########### test code ##################
        # ########################################

    result = eval_detection_voc(
        box, label, score, gt_bbox, gt_label, use_07_metric=True)
    return result


def train(epochs, dict_train, pretrained_model=Config.load_path):
    faster_rcnn = FasterRCNNVGG16().cuda()
    faster_rcnn.get_optimizer(Config.lr)
    print('model constructed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if pretrained_model:
        trainer.load(pretrained_model)

    for epoch in range(epochs):
        for i, [img_dir, img_info] in tqdm(enumerate(dict_train.items())):
            if len(img_info['objects']) == 0:
                continue
            img, img_info = rescale_image(img_dir, img_info)
            img_tensor = create_img_tensor(img)
            trainer.train_step(img_tensor, img_info)
            if (i + 1) % 2500 == 0:
                trainer.save('faster_rcnn_model.pt')
        if epoch == 7:
            trainer.faster_rcnn.scale_lr(Config.lr_decay)
    trainer.save('faster_rcnn_model.pt')


if __name__ == '__main__':
    from os.path import isfile
    if not (isfile('dict_train.npy') and isfile('dict_val.npy') and isfile('dict_test.npy')):
        dict_train, dict_val, dict_test = \
            generate_train_val_test_data(Config.img_box_dict)
        np.save('dict_train.npy', dict_train)
        np.save('dict_test.npy', dict_test)
        np.save('dict_val.npy', dict_val)

    dict_train = np.load('dict_train.npy')[()]
    train(epochs=Config.epoch, dict_train=dict_train)
    # #
    # dict_test = np.load('dict_test.npy')[()]
    # faster_rcnn = FasterRCNNVGG16().cuda()
    # state_dict = torch.load('fast_rcnn_model.pt')
    # faster_rcnn.load_state_dict(state_dict['model'])
    # result = evaluation(dict_test, faster_rcnn, test_num=100)
    # print(result)