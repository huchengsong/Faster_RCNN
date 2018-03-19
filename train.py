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
from eval_utils import calc_map
from configure import Config
from voc_parse_xml import voc_generate_img_box_dict


def generate_train_val_data(img_dict, p_train=0.9):
    """
    retrun training, validation, test subsample
    :param img_dict: dictionary storing image directory and labeling
    :param p_train: ratio of training images
    :return: dict_train, dict_val, dict_test
    """
    total_imgs = len(img_dict)
    num_train_imgs = int(total_imgs * p_train)
    img_dict_items = list(img_dict.items())
    np.random.shuffle(img_dict_items)
    dict_train, dict_val = \
        np.split(img_dict_items, [num_train_imgs])
    dict_train, dict_val = dict(dict_train), dict(dict_val)
    return dict_train, dict_val


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
    """
    return mean average precision
    :param eval_dict: dictionary with information of images to be evaluated
    :param faster_rcnn: trained faster rcnn model
    :param test_num: the number of images to be tested
    :return: mean average precision
    """
    bboxes, labels, scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for i, [img_dir, img_info] in tqdm(enumerate(eval_dict.items())):
        if len(img_info['objects']) == 0:
            continue
        img, img_info = rescale_image(img_dir, img_info, flip=False)
        img_tensor = create_img_tensor(img)
        box, score, label = faster_rcnn.predict(img_tensor)

        gt_bbox = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
        gt_label = np.array(img_info['objects'])[:, 0]
        gt_label = text_to_num(gt_label)
        bboxes.append(box)
        labels.append(label)
        scores.append(score)
        gt_bboxes.append(gt_bbox)
        gt_labels.append(gt_label)
        if i == test_num:
            break

        """
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
        """
    AP, mAP = calc_map(bboxes, labels, scores, gt_bboxes, gt_labels, use_07_metric=True)
    return mAP


def train(epochs, img_box_dict, pretrained_model=Config.load_path):
    faster_rcnn = FasterRCNNVGG16().cuda()
    faster_rcnn.get_optimizer(Config.lr)
    print('model constructed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if pretrained_model:
        trainer.load(pretrained_model)

    max_map = 0
    for epoch in range(epochs):
        print('epoch: ', epoch)
        # randomly divide data into training and validation subset for each epoch
        dict_train, dict_val = generate_train_val_data(img_box_dict)

        for i, [img_dir, img_info] in tqdm(enumerate(dict_train.items())):
            if len(img_info['objects']) == 0:
                continue
            img, img_info = rescale_image(img_dir, img_info, flip=True)
            img_tensor = create_img_tensor(img)
            trainer.train_step(img_tensor, img_info)

        # save the model with better evaluation result
        map = evaluation(dict_val, faster_rcnn, test_num=Config.eval_num)
        print('mAP: ', map, 'max mAP: ', max_map)
        if map > max_map:
            max_map = map
            trainer.save('faster_rcnn_model.pt')

        # lr decay
        if epoch == 2 or epoch == 9:
            trainer.faster_rcnn.scale_lr(Config.lr_decay)


if __name__ == '__main__':
    xml_dir = '../VOCdevkit2007/VOC2007/Annotations'
    img_dir = '../VOCdevkit2007/VOC2007/JPEGImages'
    img_box_dict = voc_generate_img_box_dict(xml_dir, img_dir)
    train(14, img_box_dict)

    xml_dir = '../VOCtest2007/VOC2007/Annotations'
    img_dir = '../VOCtest2007/VOC2007/JPEGImages'
    test_dict = voc_generate_img_box_dict(xml_dir, img_dir)
    faster_rcnn = FasterRCNNVGG16().cuda()
    state_dict = torch.load('faster_rcnn_model.pt')
    faster_rcnn.load_state_dict(state_dict['model'])
    mAP = evaluation(test_dict, faster_rcnn)
    print(mAP)
