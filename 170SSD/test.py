import sys
from PIL import Image, ImageDraw
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import EzDetectConfig
from model import EzDetectNet

from bbox import decodeAllBox, doNMS

ezConfig = EzDetectConfig()
ezConfig.batchSize = 1

print(sys.argv[1])
print(sys.argv[2])

mymodel = EzDetectNet(ezConfig, True)
mymodel.load_state_dict(torch.load(sys.argv[1]))
print("finish load model")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformer = transforms.Compose([transforms.ToTensor(),normalize])

img = Image.open(sys.argv[2])
originImage = img

img = img.resize((ezConfig.targetWidth, ezConfig.targetHeight), Image.BILINEAR)
img = transformer(img)
img = img*256
img = img.view(1, 3, ezConfig.targetHeight, ezConfig.targetWidth)
print("finish preprocess image")

img = img.cuda()
mymodel.cuda()

classOut, bboxOut = mymodel(Variable(img))
bboxOut = bboxOut.float()
bboxOut = decodeAllBox(ezConfig, bboxOut.data)

classScore = torch.nn.Softmax()(classOut[0])
bestBox = doNMS(ezConfig, classScore.data.float(), bboxOut[0], 0.08)

draw = ImageDraw.Draw(originImage)
imgWidth, imgHeight = originImage.size
for b in bestBox:
    draw.rectangle([b[0]*imgWidth, b[1]*imgHeight,
                    b[2]*imgWidth, b[3]*imgHeight])

del draw

print ("finish draw boxes")
originImage.save("1.jpg")
print ("finish all!")