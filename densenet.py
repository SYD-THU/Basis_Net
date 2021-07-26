# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class conv_densenet(nn.Module):  # BN - RELU - CONV
    def __init__(self, chn_in, chn_out, kernel_size, stride, padding):
        super(conv_densenet, self).__init__()
        self.bn_1 = nn.BatchNorm2d(chn_in)
        self.conv_1 = nn.Conv2d(chn_in, chn_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv_1(F.relu(self.bn_1(x)))

class Dense_Block(nn.Module):
    def __init__(self, chn_in, num_layer, growth_rate):
        super(Dense_Block, self).__init__()
        self.num_layer = num_layer
        self.chn_in = chn_in
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layer):
            layer_list.append(nn.Sequential(
                conv_densenet(self.chn_in+i*self.k, 4*self.k, kernel_size=1, stride=1, padding=0),
                conv_densenet(4*self.k, self.k, kernel_size=3, stride=1, padding=1)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        cat = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](cat)
            cat = torch.cat((cat, feature), 1)
        return cat

class DenseNet(nn.Module):
    def __init__(self, layers_in_stage, growth_rate, theta, num_classes):
        '''
        :param layers_in_stage:  每个stage中包含的层数
        :param growth_rate:      paper中的k，每层卷积后产生的feature map数量
        :param theta:
        :param num_classes:
        '''
        super(DenseNet, self).__init__()
        self.layer = layers_in_stage
        self.k = growth_rate
        self.theta = theta
        self.conv_1 = conv_densenet(3, 2*self.k, kernel_size=7, stride=2, padding=3)
        self.blocks, patches = self.__make_blocks(2*self.k)
        self.fc = nn.Linear(patches, num_classes)

    def __make_transitions(self, chn_in):
        chn_out = int(chn_in * self.theta)   # transition层中conv的输出通道数
        return nn.Sequential(
            conv_densenet(chn_in, chn_out, 1, 1, 0),
            nn.AvgPool2d(2)), chn_out

    def __make_blocks(self, chn_in): # 输出通道数靠参数传递设定
        layer_list = []
        patches = 0
        for i in range(len(self.layer)):
            layer_list.append(Dense_Block(chn_in, self.layer[i], self.k))
            patches = chn_in + self.layer[i]*self.k
            if i != len(self.layer)-1:
                transition, chn_in = self.__make_transitions(patches)
                layer_list.append(transition)
        return nn.Sequential(*layer_list), patches

    def forward(self, x):
        out = self.conv_1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.blocks(out)
        out = F.max_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out), dim=1)
        return out

if __name__ == '__main__':
    net_test_121 = DenseNet([6, 12, 24, 16], growth_rate=32, theta=0.5, num_classes=5)
    #summary(net, (3, 224, 224))
    x = torch.randn((2, 3, 224, 224))
    y = net_test_121(x)
    print(y)
    print(y.shape)









