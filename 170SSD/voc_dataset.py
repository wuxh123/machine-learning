from os import listdir  # 解析VOC数据路径时使用
from os.path import join
from random import random
from PIL import Image, ImageDraw
import xml.etree.ElementTree  # 用于解析VOC的xmllabel

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sampling import sampleEzDetect

__all__ = ["vocClassName", "vocClassID", "vocDataset"]

vocClassName = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']


def getVOCInfo(xmlFile):
    root = xml.etree.ElementTree.parse(xmlFile).getroot();
    anns = root.findall('object')

    bboxes = []
    for ann in anns:
        name = ann.find('name').text
        newAnn = {}
        newAnn['category_id'] = name

        bbox = ann.find('bndbox')
        newAnn['bbox'] = [-1, -1, -1, -1]
        newAnn['bbox'][0] = float(bbox.find('xmin').text)
        newAnn['bbox'][1] = float(bbox.find('ymin').text)
        newAnn['bbox'][2] = float(bbox.find('xmax').text)
        newAnn['bbox'][3] = float(bbox.find('ymax').text)
        bboxes.append(newAnn)

    return bboxes


class vocDataset(data.Dataset):
    def __init__(self, config, isTraining=True):
        super(vocDataset, self).__init__()
        self.isTraining = isTraining
        self.config = config

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # 用均值和方差对图片的RGB值分别进行归一化（也有其他方法，这种相对比较简单）
        self.transformer = transforms.Compose([transforms.ToTensor(), normalize])

    def __getitem__(self, index):
        item = None
        if self.isTraining:
            item = allTrainingData[index % len(allTrainingData)]
        else:
            item = allTestingData[index % len(allTestingData)]

        img = Image.open(item[0])  # item[0]为图像数据
        allBboxes = getVOCInfo(item[1])  # item[1]为通过getVOCInfo函数解析出真实label的数据
        imgWidth, imgHeight = img.size

        targetWidth = int((random() * 0.25 + 0.75) * imgWidth)
        targetHeight = int((random() * 0.25 + 0.75) * imgHeight)

        # 对图片进行随机crop，并保证bbox大小
        xmin = int(random() * (imgWidth - targetWidth))
        ymin = int(random() * (imgHeight - targetHeight))
        img = img.crop((xmin, ymin, xmin + targetWidth, ymin + targetHeight))
        img = img.resize((self.config.targetWidth, self.config.targetHeight), Image.BILINEAR)
        imgT = self.transformer(img)
        imgT = imgT * 256

        # 调整bbox
        bboxes = []
        for i in allBboxes:
            xl = i['bbox'][0] - xmin
            yt = i['bbox'][1] - ymin
            xr = i['bbox'][2] - xmin
            yb = i['bbox'][3] - ymin

            if xl < 0:
                xl = 0;
            if xr >= targetWidth:
                xr = targetWidth - 1
            if yt < 0:
                yt = 0
            if yb >= targetHeight:
                yb = targetHeight - 1

            xl = xl / targetWidth
            xr = xr / targetWidth
            yt = yt / targetHeight
            yb = yb / targetHeight

            if (xr - xl >= 0.05 and yb - yt >= 0.05):
                bbox = [vocClassID[i['category_id']],
                        xl, yt, xr, yb]

                bboxes.append(bbox)

        if len(bboxes) == 0:
            return self[index + 1]

        target = sampleEzDetect(self.config, bboxes);

        '''
        ### 对预测图片进行测试 ##########
        draw = ImageDraw.Draw(img)
        num = int(target[0])
        for j in range(0,num):
            offset = j * 6
            if ( target[offset + 1] < 0):
                break

            k = int(target[offset + 6])
            trueBox = [ target[offset + 2],
                        target[offset + 3],
                        target[offset + 4],
                        target[offset + 5] ]

            predBox = self.config.predBoxes[k]

            draw.rectangle([trueBox[0]*self.config.targetWidth,
                                        trueBox[1]*self.config.targetHeight,
                                        trueBox[2]*self.config.targetWidth,
                                        trueBox[3]*self.config.targetHeight])

            draw.rectangle([predBox[0]*self.config.targetWidth,
                                        predBox[1]*self.config.targetHeight,
                                        predBox[2]*self.config.targetWidth,
                                        predBox[3]*self.config.targetHeight], None, "red")

        del draw
        img.save("/tmp/{}.jpg".format(index) )
        '''

        return imgT, target

    def __len__(self):
        if self.isTraining:
            num = len(allTrainingData) - (len(allTrainingData) % self.config.batchSize)
            return num
        else:
            num = len(allTestingData) - (len(allTestingData) % self.config.batchSize)
            return num

vocClassID = {}
for i in range(len(vocClassName)):
    vocClassID[vocClassName[i]] = i + 1

print(vocClassID)
allTrainingData = []  # 第167行，该行后面的代码为从VOC2007中读取数据，会在调用voc_dataset.py文件时立即执行
allTestingData = []
allFloder = ["./VOCdevkit/VOC2007"]  # 我们把从VOC网站下载的数据放到本地，只使用VOC2007做实验
for floder in allFloder:
    imagePath = join(floder, "JPEGImages")
    infoPath = join(floder, "Annotations")
    index = 0

    for f in listdir(imagePath):  # 遍历9964张原始图片
        if f.endswith(".jpg"):
            imageFile = join(imagePath, f)
            infoFile = join(infoPath, f[:-4] + ".xml")
            if index % 10 == 0:  # 每10张随机抽1个样本做测试
                allTestingData.append((imageFile, infoFile))
            else:
                allTrainingData.append((imageFile, infoFile))

            index = index + 1