import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torchvision.models as models

from sampling import buildPredBoxes

__all__ = ["EzDetectConfig", "EzDetectNet","Test1"]

class Test1(object):
    def __init__(self):
        print('call init1')

class EzDetectConfig(object):
    def __init__(self, batchSize=4, gpu=False):
        super(EzDetectConfig, self).__init__()
        self.batchSize = batchSize
        self.gpu = gpu
        self.classNumber = 21
        self.targetWidth = 330
        self.targetHeight = 330
        self.featureSize = [[42, 42],       ## L2 1/8
                             [21, 21],      ## L3 1/16
                             [11, 11],      ## L4 1/32
                             [ 6,  6],      ## L5 1/64
                             [ 3,  3]]      ## L6 1/110

        ##[min, max, ratio, ]
        priorConfig = [[0.10, 0.25, 2],
                       [0.25, 0.40, 2, 3],
                       [0.40, 0.55, 2, 3],
                       [0.55, 0.70, 2, 3],
                       [0.70, 0.85, 2]]

        self.mboxes = []
        for i in range(len(priorConfig)):
            minSize = priorConfig[i][0]
            maxSize = priorConfig[i][1]
            meanSize = math.sqrt(minSize*maxSize)
            ratios = priorConfig[i][2:]

            # aspect ratio 1 for min and max
            self.mboxes.append([i, minSize, minSize])
            self.mboxes.append([i, meanSize, meanSize])

            # other aspect ratio
            for r in ratios:
                ar = math.sqrt(r)
                self.mboxes.append([i, minSize*ar, minSize/ar])
                self.mboxes.append([i, minSize/ar, minSize*ar])

        self.predBoxes = buildPredBoxes(self)

class EzDetectNet(nn.Module):
    def __init__(self, config, pretrained=False):
        super(EzDetectNet, self).__init__()

        self.config = config
        resnet = models.resnet50(pretrained) #从Pytorch的预训练库中拿到ResNet50模型，直接载入

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layer5 = nn.Sequential( #直到第5层才开始自定义，前面都直接复用ResNet50的结构
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        inChannles = [512, 1024, 2048, 1024, 512]
        self.locConvs = []
        self.confConvs = []
        for i in range(len(config.mboxes)):
            inSize = inChannles[ config.mboxes[i][0] ]
            confConv = nn.Conv2d(inSize, config.classNumber, kernel_size=3, stride=1, padding=1, bias=True)
            locConv = nn.Conv2d(inSize, 4, kernel_size=3, stride=1, padding=1, bias=True)

            self.locConvs.append(locConv)
            self.confConvs.append(confConv)

            super(EzDetectNet, self).add_module("{}_conf".format(i), confConv)
            super(EzDetectNet, self).add_module("{}_loc".format(i), locConv)

    def forward(self, x):
        batchSize = x.size()[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)

        featureSource = [l2, l3, l4, l5, l6]

        confs = []
        locs = []
        for i in range(len(self.config.mboxes)):
            x = featureSource[ self.config.mboxes[i][0] ]

            loc = self.locConvs[i](x)
            loc = loc.permute(0, 2, 3, 1)
            loc = loc.contiguous()
            loc = loc.view(batchSize, -1, 4)
            locs.append(loc)

            conf = self.confConvs[i](x)
            conf = conf.permute(0, 2, 3, 1)
            conf = conf.contiguous()
            conf = conf.view(batchSize, -1, self.config.classNumber)
            confs.append(conf)

        locResult = torch.cat(locs, 1)
        confResult = torch.cat(confs, 1)

        return confResult, locResult