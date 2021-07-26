# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_50_Basic_Block(nn.Sequential):
    # no dimension changed
    def __init__(self, chn_in, chn_out, kernels, sds, pds):
        '''
        :param chn_in:     int，输入通道数
        :param chn_out:    list, 各层输出通道数
        :param kernels:    list，各层核尺寸
        :param sds:        list，步长
        :param pds:        list，padding，padding=[0,1,2,3]
        '''
        super(ResNet_50_Basic_Block, self).__init__()
        # nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding )
        self.conv_1 = nn.Conv2d(chn_in, chn_out[0], kernels[0], sds[0], pds[0])
        self.bn_1 = nn.BatchNorm2d(chn_out[0])
        self.conv_2 = nn.Conv2d(chn_out[0], chn_out[1], kernels[1], sds[1], pds[1])
        self.bn_2 = nn.BatchNorm2d(chn_out[1])
        self.conv_3 = nn.Conv2d(chn_out[1], chn_out[2], kernels[2], sds[2], pds[2])
        self.bn_3 = nn.BatchNorm2d(chn_out[2])

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(self.bn_1(out))

        out = self.conv_2(out)
        out = F.relu(self.bn_2(out))

        out = self.conv_3(out)
        out = self.bn_3(out)

        out = F.relu(out + x)
        return out

class ResNet_50_Down_Block(nn.Sequential):
    # with dimension down
    # 每个stage中第一个block将起到'纬度匹配'作用，便于short相加
    def __init__(self,chn_in, chn_out, kernels, sds, pds):
        '''
        :param chn_in:     int，输入通道数
        :param chn_out:    list, 各层输出通道数
        :param kernels:    list，各层核尺寸
        :param sds:        list，步长  4
        :param pds:        list，padding， 3
        '''
        super(ResNet_50_Down_Block, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_1 = nn.Conv2d(chn_in, chn_out[0], kernels[0], sds[0], pds[0])
        self.bn_1 = nn.BatchNorm2d(chn_out[0])
        self.conv_2 = nn.Conv2d(chn_out[0], chn_out[1], kernels[1], sds[1], pds[1])
        self.bn_2 = nn.BatchNorm2d(chn_out[1])
        self.conv_3 = nn.Conv2d(chn_out[1], chn_out[2], kernels[2], sds[2], pds[2])
        self.bn_3 = nn.BatchNorm2d(chn_out[2])
        self.extra = nn.Sequential(
            nn.Conv2d(chn_in, chn_out[2], kernel_size=1, stride=sds[3], padding=0),
            nn.BatchNorm2d(chn_out[2])
        ) # 纬度匹配
    def forward(self, x):
        x_shortcut = self.extra(x)
        out = F.relu_(self.bn_1(self.conv_1(x)))
        out = F.relu_(self.bn_2(self.conv_2(out)))
        out = self.bn_3(self.conv_3(out))
        out = F.relu(out + x_shortcut)
        return out

class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=3) # stage_1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 64*56*56
        self.stage_2 = nn.Sequential(
            ResNet_50_Down_Block(64, chn_out=[64, 64, 256], kernels=[1, 3, 1], sds=[1, 2, 1, 2], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(256, chn_out=[64, 64, 256], kernels=[1, 3, 1], sds=[1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(256, chn_out=[64, 64, 256], kernels=[1, 3, 1], sds=[1, 1, 1], pds=[0, 1, 0]),
        )
        # 256*56*56
        self.stage_3 = nn.Sequential(
            ResNet_50_Down_Block(256, chn_out=[128, 128, 512], kernels=[1, 3, 1], sds=[1, 2, 1, 2], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(512, chn_out=[128, 128, 512], kernels=[1, 3, 1],sds=[1, 1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(512, chn_out=[128, 128, 512], kernels=[1, 3, 1], sds=[1, 1, 1, 1],pds=[0, 1, 0]),
            ResNet_50_Basic_Block(512, chn_out=[128, 128, 512], kernels=[1, 3, 1], sds=[1, 1, 1, 1],pds=[0, 1, 0]),
        )
        # 512*28*28
        self.stage_4 = nn.Sequential(
            ResNet_50_Down_Block(512, chn_out=[256, 256, 1024], kernels=[1, 3, 1], sds=[1, 2, 1, 2], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(1024, chn_out=[256, 256, 1024], kernels=[1, 3, 1], sds=[1, 1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(1024, chn_out=[256, 256, 1024], kernels=[1, 3, 1], sds=[1, 1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(1024, chn_out=[256, 256, 1024], kernels=[1, 3, 1], sds=[1, 1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(1024, chn_out=[256, 256, 1024], kernels=[1, 3, 1], sds=[1, 1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(1024, chn_out=[256, 256, 1024], kernels=[1, 3, 1], sds=[1, 1, 1, 1], pds=[0, 1, 0])
        )
        # 1024*14*14
        self.stage_5 = nn.Sequential(
            ResNet_50_Down_Block(1024, chn_out=[512, 512, 2048], kernels=[1, 3, 1], sds=[1, 2, 1, 2], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(2048, chn_out=[512, 512, 2048], kernels=[1, 3, 1], sds=[1, 1, 1], pds=[0, 1, 0]),
            ResNet_50_Basic_Block(2048, chn_out=[512, 512, 2048], kernels=[1, 3, 1], sds=[1, 1, 1], pds=[0, 1, 0]),
        )
        # 2048*7*7
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 10)
    def forward(self,x):
        out = self.conv_1(x)
        out = self.maxpool(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)
        out = self.stage_5(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(x.shape[0], -1))
        print('results:')
        return out

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    print('x_shape: ', x.shape)
    net = ResNet_50()
    out = net(x)
    print('out_shape: ', out.shape)
    print(out)








