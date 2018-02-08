import cv2
import numpy as np
import torch
import os
import urllib.request
import tarfile

from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable

from rescale_image import rescale_image
from fast_rcnn_model import Fast_RCNN
from voc2012_parse_xml import voc2012_generate_img_box_dict
from fast_rcnn_utils import generate_loss
from fast_rcnn_utils import generate_gt

MODEL_NAME = 'vgg16'


def train_fast_rcnn(img_dict_dir, epoch=1, cuda=False):
    img_dict = np.load(img_dict_dir)[()]
    img_num = len(img_dict)
    img_index = 0
    net = Fast_RCNN(MODEL_NAME)
    if cuda == 1:
        net.cuda()
    optimizer = optim.Adam(net.parameters())
    print(net)

    for i in range(epoch):
        for img_dir, img_info in img_dict.items():
            img, img_info = rescale_image('../' + img_dir, img_info)
            # normalize
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = Variable(normalize(torch.from_numpy(np.transpose(img / 255, (2, 0, 1))))).float()
            img = torch.unsqueeze(img, 0)
            if cuda == 1:
                img = img.cuda()
            region_proposals = generate_region_proposals(img_info)
            print(len(region_proposals))
            class_pred, bbox_pred = net(img, img_info, region_proposals)
            label_gt, bbox_gt = generate_gt(img_info, region_proposals)
            loss = generate_loss(class_pred, bbox_pred, label_gt, bbox_gt, cuda)

            print('image: {}/{}, loss: {}'.format(img_index, img_num, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            img_index = img_index+1
        img_index = 0

        file_path = os.path.join("rpn_trained.pkl")
        torch.save(net, file_path)


def generate_region_proposals(img_info):
    from generate_anchors import generate_anchors
    H, W = img_info['img_size']
    base_size = 16
    ratios = [0.5, 1.0, 2.0]
    scales = [8, 16, 32]
    anchor_list = generate_anchors(base_size, ratios, scales)
    anchor_x_shift = np.arange(0, int(W/64)) * base_size + base_size / 2
    anchor_y_shift = np.arange(0, int(H/64)) * base_size + base_size / 2
    anchor_centers = np.array([[i, j]
                               for i in anchor_y_shift
                               for j in anchor_x_shift])
    all_anchors = np.array([np.tile(anchor_centers[i], 2) + anchor_list[j]
                            for i in range(0, anchor_centers.shape[0])
                            for j in range(0, anchor_list.shape[0])])
    all_anchors[:, 0:2] = np.maximum(all_anchors[:, 0:2], 0)
    all_anchors[:, 2] = np.maximum(all_anchors[:, 2], H)
    all_anchors[:, 3] = np.maximum(all_anchors[:, 3], W)
    return all_anchors

def show_result(img_info, image):
    """
    :param img_info: image info dict
    :param image:
    :return:
    """
    # draw ground truth boxes on image
    for object in img_info['objects']:
        ymin, xmin, ymax, xmax = [int(i) for i in object[1:5]]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(image,
                    object[0],
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if not os.path.isdir('../../VOCdevkit'):
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

        urllib.request.urlretrieve(url, '../../VOCtrainval_11-May-2012.tar', progress)
        print('done')

        # extract files
        print('extracting...')
        tar = tarfile.open('../../VOCtrainval_11-May-2012.tar')
        tar.extractall('../../')
        tar.close()
        print('done')

    if not os.path.isfile('../../VOCdevkit/img_box_dict.npy'):
        # create image bounding box ground truth dictionary
        xml_dir = '../../VOCdevkit/VOC2012/Annotations'
        img_dir = '../../VOCdevkit/VOC2012/JPEGImages'
        save_dir = '../../VOCdevkit/img_box_dict.npy'
        img_box_dict = voc2012_generate_img_box_dict(xml_dir, img_dir)
        np.save(save_dir, img_box_dict)

    train_fast_rcnn('../../VOCdevkit/img_box_dict.npy', epoch=3, cuda=torch.cuda.is_available())


