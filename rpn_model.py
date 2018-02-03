import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

INPUT_CHANNELS = 3


class RPN(nn.Module):
    def __init__(self, model_name, anchor_num):
        super(RPN, self).__init__()
        # load pre-trained model and remove the last layer
        # Note: weights not initialized at described in the Faster R-CNN paper
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            modules = list(self.model.children())[:-2]
            self.model = nn.Sequential(*modules)

            for param in self.model.parameters():
                param.requires_grad = True

            # prediction head for classes and bounding boxes
            self.conv_shared = nn.Conv2d(2048, 512,
                                         (3, 3), stride=1,
                                         padding=1, bias=True)
            self.conv_reg = nn.Conv2d(512, 4*anchor_num, (1, 1),
                                      stride=1, padding=0, bias=True)
            self.conv_cls = nn.Conv2d(512, 2*anchor_num, (1, 1),
                                      stride=1, padding=0, bias=True)

        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.features = nn.Sequential(*list(self.model.features)[:-1])
            self.model = nn.Sequential(*list(self.model.children())[:-1])

            for param in self.model.parameters():
                param.requires_grad = True

            # prediction head for classes and bounding boxes
            self.conv_shared = nn.Conv2d(512, 512,
                                         (3, 3), stride=1,
                                         padding=1, bias=True)
            self.conv_reg = nn.Conv2d(512, 4*anchor_num, (1, 1),
                                      stride=1, padding=0, bias=True)
            self.conv_cls = nn.Conv2d(512, 2*anchor_num, (1, 1),
                                      stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.model(x)
        conv_shared = self.conv_shared(x)
        conv_shared = F.relu(conv_shared)
        conv_reg = self.conv_reg(conv_shared)
        conv_cls = self.conv_cls(conv_shared)

        return conv_cls, conv_reg


if __name__ == "__main__":
    # Create network
    net = RPN('resnet50')
    print(net)
