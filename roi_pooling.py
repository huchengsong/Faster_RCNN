import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def roi_pooling(img_info, region_proposals, feature_map, output_size=[7, 7]):
    """
    max pool region of interest
    :param img_info: dictionary with key ('img_size', 'objects')
    :param region_proposals: (N, 4)
    :param feature_map: (1, C, H, W), pytorch variable
    :param output_size: output size after pooling
    :return:
    """
    img_size = img_info['img_size']
    feature_map_H_W = feature_map.size()[2:4]
    region_proposals_norm = region_proposals / np.tile(img_size, 2)
    region_proposals_in_map = np.round(region_proposals_norm * np.tile(feature_map_H_W, 2)).astype(np.int)

    # max pooling for each region proposals
    maxpool = nn.AdaptiveMaxPool2d((output_size[0], output_size[1]))
    output = []

    for i in range(region_proposals.shape[0]):
        result = maxpool(feature_map[
                    :,
                    :,
                    region_proposals_in_map[i, 0]:region_proposals_in_map[i, 2] + 1,
                    region_proposals_in_map[i, 1]:region_proposals_in_map[i, 3] + 1])
        output.append(result)
    return torch.cat(output, 0)


if __name__ == '__main__':
    feature_map = Variable(torch.rand(1, 1, 50, 40), requires_grad=True)
    img_info = {'img_size': [800, 640]}
    region_proposals = np.array([[100, 200, 300, 400], [300, 300, 500, 500]])
    roi_pool_result = roi_pooling(img_info, region_proposals, feature_map)
    print(feature_map)
    print(roi_pool_result)

    feature_map = np.array([[0, 0, 0, 0], [0, 10, 0, 15], [0, 15, 0, 0], [0, 0, 15, 0]]).astype(np.float)
    feature_map = np.expand_dims(np.expand_dims(feature_map, axis=0), axis=0)
    feature_map = Variable(torch.from_numpy(feature_map))
    img_info = {'img_size': [400, 400]}
    region_proposals = np.array([[0, 0, 400, 400], [0, 0, 200, 200], [0, 0, 300, 300]])
    roi_pool_result = roi_pooling(img_info, region_proposals, feature_map)
    print(feature_map)
    print(roi_pool_result)
