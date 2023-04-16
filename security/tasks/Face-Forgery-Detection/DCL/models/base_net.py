import numpy as np
import timm
from timm.models.layers import SelectAdaptivePool2d
from timm.models.layers import BatchNormAct2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision import transforms
import cv2
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

__all__ = ['BinaryClassifier']

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)

class BinaryClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, drop_rate=0.2, has_feature=False, pretrained=False, **kwargs) -> None:
        """Base binary classifier
        Args:
            encoder ([nn.Module]): Backbone of the DCL
            num_classes (int, optional): Defaults to 2.
            drop_rate (float, optional):  Defaults to 0.2.
            has_feature (bool, optional): Wthether to return feature maps. Defaults to False.
            pretrained (bool, optional): Whether to use a pretrained model. Defaults to False.
        """
        super().__init__()
        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            # self.num_features = 1792
            self.num_features = self.encoder.last_channel

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.has_feature = has_feature
        self.feature_squeeze = nn.Conv2d(self.num_features, 1, 1)
        # for block1
        self.feature_squeeze2 = nn.Conv2d(32, 1, 1)
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x, inter=False):
        featuremap, inter_featmap = self.encoder.forward_features(x)
        x = self.global_pool(featuremap).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.has_feature:
            return x, featuremap, inter_featmap

        return x

if __name__ == '__main__':
    torch.set_printoptions( threshold = np.inf )
    torch.manual_seed(1234)
    name = "tf_efficientnet_b4_ns"
    device = 'cuda:0'

    model = BinaryClassifier(name, has_feature=True, pretrained=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = torch.rand(4, 3, 224, 224)
        inputs = inputs.to(device)
        out, _, __ = model(inputs)
        print(out.shape, _.shape, __.shape)
