import cv2
import numpy as np
import random
from copy import deepcopy

from configure import Config


def rescale_image(img_dir, img_info, flip=True):
    """
    rescale image such that the shorter side is s = 600 pixels
    :param img_dir: directory of a image
    :param img_info: img info of the image
    :param flip: whether flip the image
    :return: rescaled_image
    :return: modified_img_info: modified image info
    """
    img_info = deepcopy(img_info)
    min_size = Config.img_min_size
    max_size = Config.img_max_size
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = img_info['img_size'][0]
    width = img_info['img_size'][1]
    if height >= width:
        scale_ratio = min_size / width
        width_rescaled = min_size
        height_rescaled = int(height * scale_ratio)
        if height_rescaled > max_size:
            scale_ratio = max_size / height
            height_rescaled = max_size
            width_rescaled = int(width * scale_ratio)
        rescaled_img = cv2.resize(image, (width_rescaled, height_rescaled))

    if height < width:
        scale_ratio = min_size / height
        height_rescaled = min_size
        width_rescaled = int(width * scale_ratio)
        if width_rescaled > max_size:
            scale_ratio = max_size / width
            width_rescaled = max_size
            height_rescaled = int(height * scale_ratio)
        rescaled_img = cv2.resize(image, (width_rescaled, height_rescaled))

    img_info['img_size'] = [height_rescaled, width_rescaled]
    for object in img_info['objects']:
        object[1:5] = (np.array(object[1:5]) * scale_ratio).tolist()

    # randomly flip x axis
    if flip:
        rescaled_img, img_info = img_flip(rescaled_img, img_info,
                                      x_flip=random.choice([True, False]),
                                      y_flip=False)

    return rescaled_img, img_info


def img_flip(img, img_info, x_flip=False, y_flip=False):
    bbox = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
    if x_flip:
        img = cv2.flip(img, 1)
        xmax = img_info['img_size'][1] - bbox[:, 1]
        xmin = img_info['img_size'][1] - bbox[:, 3]
        bbox[:, 1] = xmin
        bbox[:, 3] = xmax
    if y_flip:
        img = cv2.flip(img, 0)
        ymax = img_info['img_size'][0] - bbox[:, 0]
        ymin = img_info['img_size'][0] - bbox[:, 2]
        bbox[:, 0] = ymin
        bbox[:, 2] = ymax
    for i in range(len(img_info['objects'])):
        img_info['objects'][i][1:5] = bbox[i].tolist()
    return img, img_info


if __name__ == '__main__':
    from train import voc_generate_img_box_dict
    from train import generate_train_val_data
    xml_dir = '../VOCdevkit2007/VOC2007/Annotations'
    img_dir = '../VOCdevkit2007/VOC2007/JPEGImages'
    img_box_dict = voc_generate_img_box_dict(xml_dir, img_dir)
    dict_train, dict_val = generate_train_val_data(img_box_dict)
    for img_dir, img_info in dict_train.items():
        print(img_info)
        image, image_info = rescale_image(img_dir, img_info, flip=True)
        print(img_info)
        print(image_info)
        for object in image_info['objects']:
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
