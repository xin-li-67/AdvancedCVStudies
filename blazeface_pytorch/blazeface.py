import torch
import torch.nn as nn

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]

        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            # 5*5 dw conv
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            # 1*1 conv
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)

        return out

class DoubleBlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]

        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            # 5*5 dw conv
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            # 1*1 conv
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            # activation
            nn.ReLU(inplace=True),
            # 5*5 dw conv
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(mid_channels),
            # 1*1 conv
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def froward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)

        return out

class BlazeFace(nn.Module):
    def __init__(self):
        super(BlazeFace, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.blazeBlocks = nn.Sequential(
            BlazeBlock(in_channels=24, out_channels=24),
            BlazeBlock(in_channels=24, out_channels=24),
            BlazeBlock(in_channels=24, out_channels=48, stride=2),
            BlazeBlock(in_channels=48, out_channels=48),
            BlazeBlock(in_channels=48, out_channels=48)
        )

        self.doubleBlazeBlocks = nn.Sequential(
            DoubleBlazeBlock(in_channels=48, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24)
        )

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blazeBlocks(x)
        x = self.doubleBlazeBlocks(x)

        return x

if __name__ == '__main__':
    model = BlazeFace()
    print(model)
