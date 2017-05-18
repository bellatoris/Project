import torch
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


class EncDec(nn.Module):
    def __init__(self):
        super(EncDec, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # input size = 6 x 256 x 192
        # conv1's output size = 32 x 128 x 96
        self.conv1 = nn.Conv2d(6, 32, kernel_size=9, stride=2, padding=4,
                               bias=False)
        self.shortcut_conv1 = conv3x3(32, 32)

        # conv2's output size = 64 x 64 x 48
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        # conv3's output size = 64 x 64 x 48
        self.conv3 = conv3x3(64, 64, stride=1)
        self.shortcut_conv2 = conv3x3(64, 64)

        # conv4 and conv5's output size = 128 x 32 x 24
        self.conv4 = conv3x3(64, 128, stride=2)
        self.conv5 = conv3x3(128, 128, stride=1)
        self.shortcut_conv3 = conv3x3(128, 128)

        # conv6's and conv7's output size = 256 x 16 x 12
        self.conv6 = conv3x3(128, 256, stride=2)
        self.conv7 = conv3x3(256, 256, stride=1)
        self.shortcut_conv4 = conv3x3(256, 256)

        # conv8's and conv 9's output size = 512 x 8 x 6
        self.conv8 = conv3x3(256, 512, stride=2)
        self.conv9 = conv3x3(512, 512, stride=1)

        # convolution and fully connected layer for pose
        self.pose_conv = conv3x3(512, 128)
        self.fc1 = nn.Linear(128 * 8 * 6, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 7)

        # up_conv1's output size = 256 x 16 x 12
        self.up_conv1 = up_conv4x4(512, 256, stride=2)

        # up_conv2's output size = 128 x 32 x 24
        self.up_conv2 = up_conv4x4(256, 128, stride=2)

        # up_conv3's output size = 64 x 64 x 48
        self.up_conv3 = up_conv4x4(128, 64, stride=2)

        # up_conv4's output size = 32 x 128 x 96
        self.up_conv4 = up_conv4x4(64, 32, stride=2)

        # up_conv5's output size = 16 x 256 x 192
        self.up_conv5 = up_conv4x4(32, 16, stride=2)

        # conv10's output size = 1 x 256 x 192
        self.conv10 = nn.Conv2d(16, 1, kernel_size=9, stride=1, padding=4,
                               bias=False)

        # Freeze those weights
        for param in self.parameters():
            param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        shortcut1 = out    # shortcut1's size = 32 x 128 x 96

        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)

        shortcut2 = out    # shortcut2's size = 64 x 64 x 48

        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)

        shortcut3 = out    # shortcut3's size = 128 x 32 x 24

        out = self.conv6(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.relu(out)

        shortcut4 = out    # shortcut4's size = 256 x 16 x 12

        out = self.conv8(out)
        out = self.relu(out)

        shortcut5 = out    # shortcut5's size = 512 x 8 x 6

        out = self.conv9(out)       # out's size = 512 x 8 x 6
        out += shortcut5
        out = self.relu(out)

        pose = self.pose_conv(out)
        pose = pose.view(pose.size(0), -1)
        pose = self.fc1(pose)
        pose = self.fc2(pose)
        pose = self.fc3(pose)

        out = self.up_conv1(out)    # out's size = 256 x 16 x 12
        out += shortcut4
        out += self.shortcut_conv4(shortcut4)
        out = self.relu(out)

        out = self.up_conv2(out)    # out's size = 128 x 32 x 24
        out += shortcut3
        out += self.shortcut_conv3(shortcut3)
        out = self.relu(out)

        out = self.up_conv3(out)    # out's size = 64 x 64 x 48
        out += shortcut2
        out += self.shortcut_conv2(shortcut2)
        out = self.relu(out)

        out = self.up_conv4(out)    # out's size = 32 x 128 x 96
        out += shortcut1
        out += self.shortcut_conv1(shortcut1)
        out = self.relu(out)

        out = self.up_conv5(out)    # out's size = 16 x 256 x 192
        out = self.relu(out)

        out = self.conv10(out)      # out's size = 1 x 256 x 192

        return out, pose

