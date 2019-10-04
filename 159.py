import torch
import numpy as np
import operator
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def bboxIOU (bboxA, bboxB):
    A_xmin = bboxA[0]
    A_ymin = bboxA[1]
    A_xmax = bboxA[2]
    A_ymax = bboxA[3]
    A_width = A_xmax - A_xmin
    A_height = A_ymax - A_ymin

    B_xmin = bboxB[0]
    B_ymin = bboxB[1]
    B_xmax = bboxB[2]
    B_ymax = bboxB[3]
    B_width = B_xmax - B_xmin
    B_height = B_ymax - B_ymin

    xmin = min(A_xmin, B_xmin)
    ymin = min(A_ymin, B_ymin)
    xmax = max(A_xmax, B_xmax)
    ymax = max(A_ymax, B_ymax)

    A_width_and = (A_width + B_width) - (xmax - xmin) #宽的交集
    A_height_and = (A_height + B_height) - (ymax - ymin) #高的交集

    if ( A_width_and <= 0.0001 or A_height_and <= 0.0001):
        return 0

    area_and = (A_width_and * A_height_and)
    area_or = (A_width * A_height) + (B_width * B_height)
    IOU = area_and / (area_or - area_and)

    return IOU