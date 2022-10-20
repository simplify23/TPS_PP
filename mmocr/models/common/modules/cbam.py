import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // 16),
                                nn.ReLU(),
                                nn.Linear(in_planes // 16, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(CBAM, self).__init__()
        self.linear = nn.Linear(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inplanes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        # self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x

        out = self.linear(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.linear2(out)
        # out = self.bn2(out)
        out = self.sa(out) * out
        out = self.ca(out) * out
        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out.transpose(1, 2)