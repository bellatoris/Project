import math
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def up_conv4x4(in_planes, out_planes, stride=2):
    "4x4 upconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                              stride=stride, padding=1, bias=False)

class ConvBatchRelu(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvBatchRelu, self).__init__()

        self.conv = conv3x3(in_planes, out_planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class EncDec(nn.Module):
    def __init__(self):
        super(EncDec, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # input size = 6 x 256 x 192
        # conv1's output size = 32 x 128 x 96
        self.cbr1 = ConvBatchRelu(6, 32, 2)
        self.cbr2 = ConvBatchRelu(32, 32, 1)
        self.cbr3 = ConvBatchRelu(32, 32, 1)
        self.cbr4 = ConvBatchRelu(32, 32, 1)
        self.shortcut_conv1 = conv3x3(32, 32)

        # conv2's output size = 64 x 64 x 48
        self.cbr5 = ConvBatchRelu(64, 64, 2)
        self.cbr6 = ConvBatchRelu(64, 64, 1)
        self.cbr7 = ConvBatchRelu(64, 64, 1)
        self.cbr8 = ConvBatchRelu(64, 64, 1)

        # conv3's output size = 64 x 64 x 48
        self.conv3 = conv3x3(64, 64, stride=1)
        self.shortcut_conv2 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)

        # conv4 and conv5's output size = 128 x 32 x 24
        self.conv4 = conv3x3(64, 128, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = conv3x3(128, 128, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.shortcut_conv3 = conv3x3(128, 128)

        # conv6's and conv7's output size = 256 x 16 x 12
        self.conv6 = conv3x3(128, 256, stride=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = conv3x3(256, 256, stride=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.shortcut_conv4 = conv3x3(256, 256)

        # conv8's and conv 9's output size = 512 x 8 x 6
        self.conv8 = conv3x3(256, 512, stride=2)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = conv3x3(512, 512, stride=1)
        self.bn9 = nn.BatchNorm2d(512)

        # convolution and fully connected layer for pose
        self.pose_conv = conv3x3(512, 128)
        self.fc1 = nn.Linear(128 * 8 * 6, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 7)

        # up_conv1's output size = 256 x 16 x 12
        self.conv10 = conv3x3(512, 256, stride=1)
        self.bn10 = nn.BatchNorm2d(256)

        # up_conv2's output size = 128 x 32 x 24
        self.conv11 = conv3x3(256, 128, stride=1)
        self.bn11 = nn.BatchNorm2d(128)

        # up_conv3's output size = 64 x 64 x 48
        self.conv12 = conv3x3(128, 64, stride=1)
        self.bn12 = nn.BatchNorm2d(64)

        # up_conv4's output size = 32 x 128 x 96
        self.conv13 = conv3x3(64, 32, stride=1)
        self.bn13 = nn.BatchNorm2d(32)

        # up_conv5's output size = 16 x 256 x 192
        self.conv14 = conv3x3(32, 16, stride=1)
        self.bn14 = nn.BatchNorm2d(16)

        # conv10's output size = 16 x 256 x 192
        self.conv15 = conv3x3(16, 16, stride=1)
        self.bn15 = nn.BatchNorm2d(16)

        self.conv16 = conv3x3(16, 16, stride=1)
        self.bn16 = nn.BatchNorm2d(16)

        self.conv17 = conv3x3(16, 16, stride=1)
        self.bn17 = nn.BatchNorm2d(16)

        self.conv18 = conv3x3(16, 1, stride=1)

        self.sigmoid = nn.Sigmoid()

        # Freeze those weights
        # for param in self.parameters():
        #     param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.cbr1(x)
        out = self.cbr2(out)
        out = self.cbr3(out)
        out = self.cbr4(out)

        shortcut1 = out    # shortcut1's size = 32 x 128 x 96

        out = self.cbr5(out)
        out = self.cbr6(out)
        out = self.cbr7(out)
        out = self.cbr8(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        shortcut2 = out    # shortcut2's size = 64 x 64 x 48

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        shortcut3 = out    # shortcut3's size = 128 x 32 x 24

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu(out)

        shortcut4 = out    # shortcut4's size = 256 x 16 x 12

        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)

        shortcut5 = out    # shortcut5's size = 512 x 8 x 6

        out = self.conv9(out)       # out's size = 512 x 8 x 6
        out = self.bn9(out)
        out += shortcut5
        out = self.relu(out)

        pose = self.pose_conv(out)
        pose = pose.reshape(pose.size(0), -1)
        pose = self.fc1(pose)
        pose = self.fc2(pose)
        pose = self.fc3(pose)

        out = self.upsample(out)
        out = self.conv10(out)    # out's size = 256 x 16 x 12
        out = self.bn10(out)
        out += shortcut4
        out += self.shortcut_conv4(shortcut4)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv11(out)    # out's size = 128 x 32 x 24
        out = self.bn11(out)
        out += shortcut3
        out += self.shortcut_conv3(shortcut3)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv12(out)    # out's size = 64 x 64 x 48
        out = self.bn12(out)
        out += shortcut2
        out += self.shortcut_conv2(shortcut2)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv13(out)    # out's size = 32 x 128 x 96
        out = self.bn13(out)
        out += shortcut1
        out += self.shortcut_conv1(shortcut1)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv14(out)    # out's size = 16 x 256 x 192
        out = self.bn14(out)
        out = self.relu(out)

        out = self.conv15(out)    # out's size = 16 x 256 x 192
        out = self.bn15(out)
        out = self.relu(out)

        out = self.conv16(out)    # out's size = 16 x 256 x 192
        out = self.bn16(out)
        out = self.relu(out)

        out = self.conv17(out)    # out's size = 16 x 256 x 192
        out = self.bn17(out)
        out = self.relu(out)

        out = self.conv18(out)    # out's size = 1 x 256 x 192

        out = self.sigmoid(out)

        return out, pose

