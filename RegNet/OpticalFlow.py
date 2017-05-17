import torch.nn as nn
import torch
import math


def up_conv4x4(in_planes, out_planes, stride=2):
    "4x4 upconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                              stride=stride, padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with no padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, padding=0, bias=False)

class OpticalFlow(nn.Module):
    def __init__(self):
        super(OpticalFlow, self).__init__()
        self.fc1 = nn.Linear(6, 480)
        self.relu = nn.ReLU(inplace=True)
        self.upConv1 = up_conv4x4(10, 10)    # input: (8, 6), output: (16, 12)
        self.upConv2 = up_conv4x4(10, 10)    # input: (16, 12), output: (32, 24)
        self.upConv3 = up_conv4x4(10, 10)    # input: (32, 24), output: (64, 48)
        self.upConv4 = up_conv4x4(10, 10)    # input: (64, 48), output: (128, 96)
        self.upConv5 = up_conv4x4(10, 10)    # input: (128, 96), output: (256, 192)
        self.conv1 = conv1x1(10, 10)
        self.conv4 = conv1x1(10, 2)
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, depth):
        out = self.fc1(x)
        out = self.relu(out)
        out = out.view(out.size(0), 10, 8, 6)
        out = self.upConv1(out)
        out = self.relu(out)
        out = self.upConv2(out)
        out = self.relu(out)
        out = self.upConv3(out)
        out = self.relu(out)
        out = self.upConv4(out)
        out = self.relu(out)
        out = self.upConv5(out)
        out = self.relu(out)

        depth = torch.cat((depth, depth, depth, depth,
                           depth, depth, depth, depth,
                           depth, depth), dim=1)

        out = torch.mul(out, depth)

        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv4(out)

        return out
