import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

from bbox import bboxIOU

__all__ = ["buildPredBoxes", "sampleEzDetect"]

def buildPredBoxes(config):
    predBoxes = []

    for i in range(len(config.mboxes)):
        l = config.mboxes[i][0]
        wid = config.featureSize[l][0]
        hei = config.featureSize[l][1]

        wbox = config.mboxes[i][1]
        hbox = config.mboxes[i][2]

        for y in range(hei):
            for x in range(wid):
                xc = (x + 0.5) / wid #x y位置都取每个feature map像素的中心点来计算
                yc = (y + 0.5) / hei
                '''
                xmin = max(0, xc-wbox/2)
                ymin = max(0, yc-hbox/2)
                xmax = min(1, xc+wbox/2)
                ymax = min(1, yc+hbox/2)
                '''
                xmin = xc-wbox/2
                ymin = yc-hbox/2
                xmax = xc+wbox/2
                ymax = yc+hbox/2

                predBoxes.append([xmin, ymin, xmax, ymax])

    return predBoxes

def sampleEzDetect(config, bboxes): #在voc_dataset.py的vocDataset类中用到的sampleEzDetect函数
    ## preparing pred boxes
    predBoxes = config.predBoxes

    ## preparing groud truth
    truthBoxes = []
    for i in range(len(bboxes)):
        truthBoxes.append( [bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]] )

    ## computing iou
    iouMatrix = []
    for i in predBoxes:
        ious = []
        for j in truthBoxes:
            ious.append( bboxIOU(i, j) )
        iouMatrix.append(ious)

    iouMatrix = torch.FloatTensor( iouMatrix )
    iouMatrix2 = iouMatrix.clone()

    ii = 0
    selectedSamples = torch.FloatTensor(128*1024)

    ## positive samples from bi-direction match
    for i in range(len(bboxes)):
        iouViewer = iouMatrix.view(-1)
        iouValues, iouSequence = torch.max(iouViewer, 0)

        predIndex = iouSequence // len(bboxes)
        bboxIndex = iouSequence % len(bboxes)

        if ( iouValues > 0.1):
            selectedSamples[ii*6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii*6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii*6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii*6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii*6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii*6 + 6] = predIndex
            ii  = ii + 1
        else:
            break

        iouMatrix[:, bboxIndex] = -1
        iouMatrix[predIndex, :] = -1
        iouMatrix2[predIndex,:] = -1

    ## also samples with high iou
    for i in range(len(predBoxes)):
        v,_ = iouMatrix2[i].max(0)
        predIndex = i
        bboxIndex = _

        if ( v > 0.7): #anchor与真实值iou大于0.7的为正样本
            selectedSamples[ii*6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii*6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii*6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii*6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii*6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii*6 + 6] = predIndex
            ii  = ii + 1

        elif (v > 0.5):
            selectedSamples[ii*6 + 1] = bboxes[bboxIndex][0] * -1
            selectedSamples[ii*6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii*6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii*6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii*6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii*6 + 6] = predIndex
            ii  = ii + 1

    selectedSamples[0] = ii
    return selectedSamples