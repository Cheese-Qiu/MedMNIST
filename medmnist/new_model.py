import torch
from torch import nn
from torchvision import models

'''
    inplane:输入通道数
    midplane：中间处理时的通道数
    midplane*extention：输出的通道数
    downsample:用来标记残差是否卷积（两种Block）
'''

class BasicBlock(nn.Module):
    extention = 1
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)
        self.conv2 = nn.Conv2d(midplane, midplane*self.extention, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane*self.extention)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差
        if (self.downsample != None):
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    extention = 4

    def __init__(self, inplane, midplane, stride=1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)
        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)
        self.conv3 = nn.Conv2d(midplane, midplane*self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midplane*self.extention)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        #残差
        if(self.downsample!=None):
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)

        return out

class RestNet(nn.Module):
        def __init__(self, block, layers, in_channels=1, num_classes=2):

            self.inplane = 64
            super(RestNet,self).__init__()
            self.block = block
            self.layers = layers

         #   self.conv1 = nn.Conv2d(in_channels, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = nn.Conv2d(in_channels, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplane)
            self.relu = nn.ReLU()
            self.Maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

            self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
            self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
            self.stage3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
            self.stage4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

            self.avgpool = nn.AvgPool2d(kernel_size=4)
            self.fc = nn.Linear(512*block.extention, num_classes)

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.Maxpool(out)

            out = self.stage1(out)
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)

            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out



        def make_layer(self, block, midplane, block_num, stride=1):
            block_list=[]
            downsample = None
            if(stride!=1 or self.inplane!=midplane*block.extention):
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplane, midplane*block.extention, stride=stride, kernel_size=1, bias=False),
                    nn.BatchNorm2d(midplane*block.extention)
                )

            #conv Block
            conv_block = block(self.inplane, midplane, stride=stride, downsample=downsample)
            block_list.append(conv_block)
            self.inplane = midplane*block.extention

            #identity Block
            for i in range(1,block_num):
                block_list.append(block(self.inplane, midplane, stride=1))

            return nn.Sequential(*block_list)

def ResNet50(in_channels, num_classes):
    return RestNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

def ResNet18(in_channels, num_classes):
    return RestNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)