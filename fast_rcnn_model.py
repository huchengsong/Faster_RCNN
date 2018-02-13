import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from roi_pooling import roi_pooling

INPUT_CHANNELS = 3
MAX_POOL_OUTPUT_SIZE = [7, 7]
CLASS_NUM = 21


class Fast_RCNN(nn.Module):
    def __init__(self, model_name):
        super(Fast_RCNN, self).__init__()
        # load pre-trained model and remove the last layer
        # Note: weights not initialized as described in the Faster R-CNN paper
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.features = nn.Sequential(*list(self.model.features)[:-1])
            self.model = nn.Sequential(*list(self.model.children())[:-1])

            for i in range(17):
                for param in self.model[0][i].parameters():
                    param.requires_grad = False

            # prediction head for classes and bounding boxes
            self.shared_fc_1 = nn.Linear(512 * MAX_POOL_OUTPUT_SIZE[0] * MAX_POOL_OUTPUT_SIZE[1], 512)
            self.shared_fc_2 = nn.Linear(512, 256)
            self.class_fc = nn.Linear(256, CLASS_NUM)
            self.bbox_fc = nn.Linear(256, CLASS_NUM * 4)
            self.softmax = nn.Softmax(dim=0)

    def forward(self, x, img_info, region_proposals):
        feature_map = self.model(x)
        roi_pooling_result = roi_pooling(img_info,
                                         region_proposals,
                                         feature_map,
                                         output_size=MAX_POOL_OUTPUT_SIZE)
        roi_pooling_result = roi_pooling_result.view((-1, 512 * MAX_POOL_OUTPUT_SIZE[0] * MAX_POOL_OUTPUT_SIZE[1]))
        fc_1 = self.shared_fc_1(roi_pooling_result)
        fc_1 = F.relu(fc_1)
        fc_2 = self.shared_fc_2(fc_1)
        fc_2 = F.relu(fc_2)
        class_pred = self.class_fc(fc_2)
        class_pred = self.softmax(class_pred)
        bbox_pred = self.bbox_fc(fc_2)

        return class_pred, bbox_pred


if __name__ == "__main__":
    # Create network
    net = Fast_RCNN('vgg16')
    print(net)



