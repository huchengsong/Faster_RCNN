import cv2
import numpy as np
import os
import urllib.request
import tarfile

from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable

from rescale_image import rescale_image
from generate_anchors import generate_anchors
from rpn_model import RPN
from voc2012_parse_xml import voc2012_generate_img_box_dict
import rpn_utils

BASE_SIZE = 16
RATIOS = [0.5, 1.0, 2.0]
SCALES = [8, 16, 32]
MODEL_NAME = 'vgg16'


def train_rpn(img_dict_dir, epoch=1, cuda=False):
    img_dict = np.load(img_dict_dir)[()]
    img_num = len(img_dict)
    img_index = 0
    net = RPN(MODEL_NAME, len(RATIOS) * len(SCALES))
    if cuda == 1:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print(net)

    for i in range(epoch):
        for img_dir, img_info in img_dict.items():
            optimizer.zero_grad()
            img, modified_img_info = rescale_image(img_dir, img_info)
            # normalize
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = Variable(normalize(torch.from_numpy(np.transpose(img / 255, (2, 0, 1))))).float()
            img = torch.unsqueeze(img, 0)
            if cuda == 1:
                img = img.cuda()
            cls_score, reg_score = net(img)
            score_dim = cls_score.size()[2:4]
            cls_gt, reg_gt = rpn_utils.generate_gt_cls_reg(modified_img_info, score_dim, BASE_SIZE, RATIOS, SCALES)

            loss = rpn_utils.generate_rpn_loss(cls_score, reg_score, cls_gt, reg_gt, cuda)
            print('image: {}/{}, loss: {}'.format(img_index, img_num, loss.data[0]))
            loss.backward()
            optimizer.step()

            img_index = img_index+1
            if img_index % 1024 == 0:
                file_path = os.path.join("rpn_trained.pkl")
                torch.save(net, file_path)
        img_index = 0
        file_path = os.path.join("rpn_trained.pkl")
        torch.save(net, file_path)


if __name__ == '__main__':
    if not os.path.isdir('../VOCdevkit'):
        url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        print('downloading VOCtrainval_11-May-2012.tar')

        import sys
        import time

        def progress(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(url, '../VOCtrainval_11-May-2012.tar', progress)
        print('done')

        # extract files
        print('extracting...')
        tar = tarfile.open('../VOCtrainval_11-May-2012.tar')
        tar.extractall('../')
        tar.close()
        print('done')

    if not os.path.isfile('../VOCdevkit/img_box_dict.npy'):
        # create image bounding box ground truth dictionary
        xml_dir = '../VOCdevkit/VOC2012/Annotations'
        img_dir = '../VOCdevkit/VOC2012/JPEGImages'
        save_dir = '../VOCdevkit/img_box_dict.npy'
        img_box_dict = voc2012_generate_img_box_dict(xml_dir, img_dir)
        np.save(save_dir, img_box_dict)

    train_rpn('../VOCdevkit/img_box_dict.npy', epoch=3, cuda=torch.cuda.is_available())


