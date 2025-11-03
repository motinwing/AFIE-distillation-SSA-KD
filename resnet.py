import torch.nn as nn
import torch
from pruneutil import *
import torch.optim as optim
import time

# 18/34
class BasicBlock(nn.Module):
    expansion = 1  # 每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN处理
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu2(out)

        return out

    def block_prune(self, threshold, index_forward, theta):
        index_r = index_forward
        index_c = conv_index(self.conv1, threshold, theta)
        conv_prune(self.conv1, index_c=index_c, index_r=index_r)
        norm_prune(self.bn1, index=index_c)
        index_r = index_c

        if self.downsample is not None:
            index_c = conv_index(self.conv2, threshold, theta)
            conv_prune(self.conv2, index_c=index_c, index_r=index_r)
            norm_prune(self.bn2, index=index_c)
            conv_prune(self.downsample[0], index_c=index_c, index_r=index_forward, ifprint=False)
            norm_prune(self.downsample[1], index=index_c)
        else:
            temp_criteria_list = taylor_criteria(self.conv2, theta)
            sorted = torch.sort(temp_criteria_list, dim=0, descending=False)[0]
            threshold = sorted[int(self.conv2.weight.size(0) - index_forward.size(0))]
            index_c = taylor_index(self.conv2, threshold, theta)
            conv_prune(self.conv2, index_c=index_c, index_r=index_r)
            norm_prune(self.bn2, index=index_c)
        print("---------------")
        return index_c

    def block_criteria(self, criteria_list, theta):
        criteria_list = conv_criteria(self.conv1, criteria_list, theta)
        criteria_list = conv_criteria(self.conv2, criteria_list, theta)
        return criteria_list


# 50,101,152
class Bottleneck(nn.Module):
    expansion = 4  # 4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,  # 输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu3(out)

        return out

    def block_prune(self, threshold, index_forward, theta):
        index_r = index_forward
        index_c = conv_index(self.conv1, threshold, theta)
        conv_prune(self.conv1, index_c=index_c, index_r=index_r)
        norm_prune(self.bn1, index=index_c)
        index_r = index_c
        index_c = conv_index(self.conv2, threshold, theta)
        conv_prune(self.conv2, index_c=index_c, index_r=index_r)
        norm_prune(self.bn2, index=index_c)
        index_r = index_c
        if self.downsample is not None:
            index_c = conv_index(self.conv3, threshold, theta)
            conv_prune(self.conv3, index_c=index_c, index_r=index_r)
            norm_prune(self.bn3, index=index_c)
            conv_prune(self.downsample[0], index_c=index_c, index_r=index_forward, ifprint=False)
            norm_prune(self.downsample[1], index=index_c)
        else:
            temp_criteria_list = taylor_criteria(self.conv3, theta)
            sorted = torch.sort(temp_criteria_list, dim=0, descending=False)[0]
            threshold = sorted[int(self.conv3.weight.size(0) - index_forward.size(0))]
            index_c = taylor_index(self.conv3, threshold, theta)
            conv_prune(self.conv3, index_c=index_c, index_r=index_r)
            norm_prune(self.bn3, index=index_c)
        print("---------------")
        return index_c

    def block_criteria(self, criteria_list, theta):
        criteria_list = conv_criteria(self.conv1, criteria_list, theta)
        criteria_list = conv_criteria(self.conv2, criteria_list, theta)
        criteria_list = conv_criteria(self.conv3, criteria_list, theta)
        return criteria_list


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):  # block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        # torch.manual_seed(int(time.time() * 1000))
        self.include_top = include_top
        self.in_channel = 64
        # self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_channel)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for param in self.parameters():
            param.requires_grad = True

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        # else:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.in_channel, channel, kernel_size=1, stride=1, bias=False),
        #         nn.BatchNorm2d(channel))

        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride)]
        self.in_channel = channel * block.expansion

        downsample = None
        for _ in range(1, block_num):
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1, stride=1, bias=False),
            #     nn.BatchNorm2d(self.in_channel))
            layers.append(block(self.in_channel, channel, downsample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

    def resnet_prune(self, global_sparsity, theta):
        # before_p = 0
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         before_p += m.weight.size(0) * m.weight.size(1) * m.weight.size(2) * m.weight.size(3)

        criteria_list = torch.zeros(0, device=self.conv1.weight.device)
        criteria_list = conv_criteria(self.conv1, criteria_list, theta)

        for block in self.layer1:
            criteria_list = block.block_criteria(criteria_list, theta)

        for block in self.layer2:
            criteria_list = block.block_criteria(criteria_list, theta)

        for block in self.layer3:
            criteria_list = block.block_criteria(criteria_list, theta)

        for block in self.layer4:
            criteria_list = block.block_criteria(criteria_list, theta)

        # sorted = torch.sort(criteria_list, dim=0, descending=False)[0]
        # threshold = sorted[int(sorted.size(0) * global_sparsity)]

        p_star = torch.sum(criteria_list[1::2], dim=0)
        threshold = global_sparsity * p_star / torch.sum(criteria_list[1::2] / criteria_list[0::2], dim=0)
        print(criteria_list[0::2])
        print("\n")
        index_c = taylor_index(self.conv1, -100, theta)
        # index_c = None
        conv_prune(self.conv1, index_c=index_c)
        norm_prune(self.bn1, index_c)

        for block in self.layer1:
            index_c = block.block_prune(threshold, index_c, theta)

        for block in self.layer2:
            index_c = block.block_prune(threshold, index_c, theta)

        for block in self.layer3:
            index_c = block.block_prune(threshold, index_c, theta)

        for block in self.layer4:
            index_c = block.block_prune(threshold, index_c, theta)

        fc_prune(self.fc, index_r=index_c)

        # after_p = 0
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         after_p += m.weight.size(0) * m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
        # prune_rate = after_p / before_p
        #
        # print("prune rate:{0}".format(prune_rate))


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
