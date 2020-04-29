import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Cifar_Very_Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Very_Tiny, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=32 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

        self.output_layer = -1

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        if self.output_layer == 0:
            return out

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        if self.output_layer == 1:
            return out

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        if self.output_layer == 2:
            return out

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.output_layer == 3:
            return out

        out = self.fc2(out)

        return out

    def get_features(self, x, layers=[-1]):

        layers = np.asarray(layers)
        features = [None] * len(layers)

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        if 0 in layers:
            # out_t = F.avg_pool2d(out, 2) #12
            out_t = out.view(out.size(0), -1)
            idx = np.where(layers == 0)[0]
            for i in idx:
                features[i] = out_t

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        if 1 in layers:
            # out_t = F.avg_pool2d(out, 2) # 6
            out_t = out.view(out.size(0), -1)
            idx = np.where(layers == 1)[0]
            for i in idx:
                features[i] = out_t

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        if 2 in layers:
            # out_t = F.avg_pool2d(out, 2)
            out_t = out.view(out.size(0), -1)
            idx = np.where(layers == 2)[0]
            for i in idx:
                features[i] = out_t

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))

        if 3 in layers:
            idx = np.where(layers == 3)[0]
            for i in idx:
                features[i] = out

        out = self.fc2(out)
        if 4 in layers:
            idx = np.where(layers == 4)[0]
            for i in idx:
                features[i] = out

        return features
