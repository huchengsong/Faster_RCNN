import cv2
import numpy as np


def rescale_image(img_dir, img_info):
    """
    rescale image such that the shorter side is s = 600 pixels
    :param img_dir: directory of a image
    :param img_info: img info of the image
    :return: rescaled_image
    :return: modified_img_info: modified image info
    """
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = img_info['img_size'][0]
    width = img_info['img_size'][1]
    if height >= width:
        scale_ratio = 600 / width
        width_rescaled = 600
        height_rescaled = int(height * scale_ratio)
        if height_rescaled > 1000:
            scale_ratio = 1000 / height
            height_rescaled = 1000
            width_rescaled = int(width * scale_ratio)
        rescaled_img = cv2.resize(image, (width_rescaled, height_rescaled))

    if height < width:
        scale_ratio = 600 / height
        height_rescaled = 600
        width_rescaled = int(width * scale_ratio)
        if width_rescaled > 1000:
            scale_ratio = 1000 / width
            width_rescaled = 1000
            height_rescaled = int(height * scale_ratio)
        rescaled_img = cv2.resize(image, (width_rescaled, height_rescaled))

    img_info['img_size'] = [height_rescaled, width_rescaled]
    for object in img_info['objects']:
        object[1:5] = (np.array(object[1:5]) * scale_ratio).tolist()

    return rescaled_img, img_info

if __name__ == '__main__':
    img_box_dict = np.load('../VOCdevkit/img_box_dict.npy')[()]
    for img_dir, img_info in img_box_dict.items():
        print(img_info)
        image, image_info = rescale_image(img_dir, img_info)
        print(img_info)
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
