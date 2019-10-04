import cv2
import argparse
import os, sys, shutil
import numpy as np
import pandas as pd
import time
import glob
from collections import defaultdict
from tqdm import tqdm

from dataset import UNetDataset

import torch
import torchvision
from torch.nn import DataParallel, CrossEntropyLoss
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

#import cfgs.config as cfg
from unet import UNet

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)
    if not os.path.exists(path):
        os.mkdir(path)

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class UNetTrainer(object):
    """UNet trainer"""
    def __init__(self, start_epoch=0, save_dir='', resume="", devices_num=2,
                 num_classes=2, color_dim=1):

        self.net = UNet(color_dim=color_dim, num_classes=num_classes)
        self.start_epoch = start_epoch if start_epoch != 0 else 1
        self.save_dir = os.path.join('../models', save_dir)
        self.loss = CrossEntropyLoss()
        self.num_classes = num_classes

        if resume:
            checkpoint = torch.load(resume)
            if self.start_epoch == 0:
                self.start_epoch = checkpoint['epoch'] + 1
            if not self.save_dir:
                self.save_dir = checkpoint['save_dir']
            self.net.load_state_dict(checkpoint['state_dir'])

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.net.cuda()
        self.loss.cuda()
        if devices_num == 2:
            self.net = DataParallel(self.net, device_ids=[0, 1])
        #self.loss = DataParallel(self.loss, device_ids=[0, 1])

    def train(self, train_loader, val_loader, lr=0.001,
              weight_decay=1e-4,
              epochs=200,
              save_freq=10):

        self.logfile = os.path.join(self.save_dir, 'log')
        sys.stdout = Logger(self.logfile)
        self.epochs = epochs
        self.lr = lr

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            #lr,
            #momentum=0.9,
            weight_decay = weight_decay)

        for epoch in range(self.start_epoch, epochs+1):
            self.train_(train_loader, epoch, optimizer, save_freq)
            self.validate_(val_loader, epoch)

    def train_(self, data_loader, epoch, optimizer, save_freq=10):
        start_time = time.time()

        self.net.train()
        #lr = self.get_lr(epoch)

        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr

        metrics = []

        for i, (data, target) in enumerate(tqdm(data_loader)):
            data_t, target_t = data, target
            data = Variable(data.cuda(non_blocking=True))
            target = Variable(target.cuda(non_blocking=True))

            output = self.net(data) #unet输出结果

            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
            target = target.view(-1)
            loss_output = self.loss(output, target)

            optimizer.zero_grad()
            loss_output.backward() #反向传播loss
            optimizer.step()

            loss_output = loss_output.data[0] #loss数值
            acc = accuracy(output, target)
            metrics.append([loss_output, acc])

            if i == 0:
                batch_size = data.size(0)
                _, output = output.data.max(dim=1)
                output = output.view(batch_size, 1, 1, 320, 480).cpu() #预测结果图
                data_t = data_t[0, 0].unsqueeze(0).unsqueeze(0) #原img图
                target_t = target_t[0].unsqueeze(0) #gt图
                t = torch.cat([output[0].float(), data_t, target_t.float()], 0) #第一个参数为list，拼接3张图像
                #show_list = []
                #for j in range(10):
                #    show_list.append(data_t[j, 0].unsqueeze(0).unsqueeze(0))
                #    show_list.append(target_t[j].unsqueeze(0))
                #    show_list.append(output[j].float())
                #
                #t = torch.cat(show_list, 0)
                torchvision.utils.save_image(t,"temp_image/%02d_train.jpg"%epoch, nrow=3)

            #if i == 20:
            #    break

        if epoch % save_freq == 0:
            if 'module' in dir(self.net):
                state_dict = self.net.module.state_dict()
            else:
                state_dict = self.net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch' : epoch,
                'save_dir' : self.save_dir,
                'state_dir' : state_dict},

                os.path.join(self.save_dir, '%03d.ckpt' % epoch))

        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        self.print_metrics(metrics, 'Train', end_time-start_time, epoch)

    def validate_(self, data_loader, epoch):
        start_time = time.time()

        self.net.eval()
        metrics = []
        for i, (data, target) in enumerate(data_loader):
            data_t, target_t = data, target
            data = Variable(data.cuda(non_blocking=True), volatile = True)
            target = Variable(target.cuda(non_blocking=True), volatile = True)

            output = self.net(data)
            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
            target = target.view(-1)
            loss_output = self.loss(output, target)

            loss_output = loss_output.data[0]
            acc = accuracy(output, target)
            metrics.append([loss_output, acc])

            if i == 0:
                batch_size = data.size(0)
                _, output = output.data.max(dim=1)
                output = output.view(batch_size, 1, 1, 320, 480).cpu()
                data_t = data_t[0, 0].unsqueeze(0).unsqueeze(0)
                target_t = target_t[0].unsqueeze(0)
                t = torch.cat([output[0].float(), data_t, target_t.float()], 0)
            #    show_list = []
            #    for j in range(10):
            #        show_list.append(data_t[j, 0].unsqueeze(0).unsqueeze(0))
            #        show_list.append(target_t[j].unsqueeze(0))
            #        show_list.append(output[j].float())
            #
            #    t = torch.cat(show_list, 0)
                torchvision.utils.save_image(t,"temp_image/%02d_val.jpg"%epoch, nrow=3)
            #if i == 10:
            #    break

        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        self.print_metrics(metrics, 'Validation', end_time-start_time)


    def print_metrics(self, metrics, phase, time, epoch=-1):
        """metrics: [loss, acc]
        """
        if epoch != -1:
            print ("Epoch: {}".format(epoch),)
        print (phase,)
        print('loss %2.4f, accuracy %2.4f, time %2.2f' % (np.mean(metrics[:, 0]), np.mean(metrics[:, 1]), time))
        if phase != 'Train':
            print

    def get_lr(self, epoch):
        if epoch <= self.epochs * 0.5:
            lr = self.lr
        elif epoch <= self.epochs * 0.8:
            lr = 0.1 * self.lr
        else:
            lr = 0.01 * self.lr
        return lr

    def save_py_files(self, path):
        """copy .py files in exps dir, cfgs dir and current dir into
           save_dir, and keep the files structure
        """
        #exps dir
        pyfiles = [f for f in os.listdir(path) if f.endswith('.py')]
        path = "/".join(path.split('/')[-2:])
        exp_save_path = os.path.join(self.save_dir, path)
        mkdir(exp_save_path)
        for f in pyfiles:
            shutil.copy(os.path.join(path, f),os.path.join(exp_save_path,f))
        #current dir
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(self.save_dir,f))
        #cfgs dir
        shutil.copytree('./cfgs', os.path.join(self.save_dir,'cfgs'))

def accuracy(output, target):
    _, pred = output.max(dim=1)
    correct = pred.eq(target)
    return correct.float().sum().data[0] / target.size(0)

class UNetTester(object):
    def __init__(self, model, devices_num=2, color_dim=1, num_classes=2):
        self.net = UNet(color_dim=color_dim, num_classes=num_classes)
        checkpoint = torch.load(model)
        self.color_dim = color_dim
        self.num_classes = num_classes
        self.net.load_state_dict(checkpoint['state_dir'])
        self.net = self.net.cuda()
        if devices_num == 2:
            self.net = DataParallel(self.net, device_ids=[0, 1])
        self.net.eval()

    def test(self, folder, target_dir):
        mkdir(target_dir)
        cracks_files = glob.glob(os.path.join(folder, "*.jpg"))
        print (len(cracks_files), "imgs.")
        for crack_file in tqdm(cracks_files):
            name = os.path.basename(crack_file)
            save_path = os.path.join(target_dir, name)

            data = cv2.imread(crack_file, cv2.IMREAD_GRAYSCALE)
            output = self._test(data) #图片结果

            cv2.imwrite(save_path, output)

    def _test(self, data):
        data = data.astype(np.float32) / 255.
        data = np.expand_dims(data, 0)
        data = np.expand_dims(data, 0)

        input = torch.from_numpy(data)
        height = input.size()[-2]
        width= input.size()[-1]
        input = Variable(input, volatile=True).cuda()
        batch_size = 1

        output = self.net(input)
        output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
        _, output = output.data.max(dim=1)
        output[output>0] = 255
        output = output.view(height, width)
        output = output.cpu().numpy()

        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crack segment')
    parser.add_argument('--train', '-t', help='train data dir', default='')
    parser.add_argument('--resume', '-r', help='the resume model path', default='')
    parser.add_argument('--wd', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--name', help='the name of the model', default='crack_segment')
    parser.add_argument('--sfreq', metavar='SF', default=10, help='model save frequency',
                        type=int)
    parser.add_argument('--test', help='test data dir', default='')
    parser.add_argument('--model', help='crack segment model path', default='')
    parser.add_argument('--target', help='target data dir', default='')
    args = parser.parse_args()
    if args.train:
        masks = glob.glob(os.path.join(args.train, 'mask/*.jpg'))
        masks.sort()
        N = len(masks)
        train_N = int(N * 0.8)
        train_loader = DataLoader(UNetDataset(mask_list=masks[:train_N], phase='train'),
                                batch_size=8, shuffle=True,
                                num_workers=32, pin_memory=True)
        val_loader = DataLoader(UNetDataset(mask_list=masks[train_N:], phase='val'),
                                batch_size=8, shuffle=True,
                                num_workers=32, pin_memory=True)
        crack_segmentor = UNetTrainer(save_dir=args.name, resume=args.resume,
                                     devices_num=cfg.devices_num)
        crack_segmentor.train(train_loader, val_loader, weight_decay=args.wd)
    if args.test:
        assert args.target, "target path must not be None."
        assert args.target, "model path must not be None."
        crack_segmentor = UNetTester(args.model, devices_num=cfg.devices_num)
        crack_segmentor.test(args.test, args.target)