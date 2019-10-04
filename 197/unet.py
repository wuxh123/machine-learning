import torch
import torch.nn as nn

def filter_for_depth(d):
    return 4*2**d

def conv2x2(in_planes, out_planes, kernel_size=3, stride=1, padding=1): #卷积层
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def upconv2x2(in_planes, out_planes, kernel_size=2, stride=2, padding=0): #反卷积层
    # Hout = (Hin-1)*stride[0] - 2*padding[0] + kernel_size[0] + output_padding[0]
    # Wout = (Win-1)*stride[1] - 2*padding[1] + kernel_size[1] + output_padding[1]

    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
                        nn.BatchNorm2d(out_planes),
                             nn.ReLU(inplace=True))

def pre_block(color_dim):
    """conv->conv"""
    in_filter = color_dim
    out_filter = filter_for_depth(0)

    layers = []
    layers.append(conv2x2(in_filter, out_filter))
    layers.append(conv2x2(out_filter, out_filter))

    return nn.Sequential(*layers)

def contraction(depth):
    """downsample->conv->conv"""
    assert depth > 0, 'depth <= 0 '

    in_filter = filter_for_depth(depth-1)
    out_filter = filter_for_depth(depth)

    layers = []
    layers.append(nn.MaxPool2d(2, stride=2))
    layers.append(conv2x2(in_filter, out_filter))
    layers.append(conv2x2(out_filter, out_filter))

    return nn.Sequential(*layers)

def upsample(depth):
    in_filter = filter_for_depth(depth)
    out_filter = filter_for_depth(depth-1)

    return nn.Sequential(upconv2x2(in_filter, out_filter))

def expansion(depth):
    """conv->conv->upsample"""

    assert depth > 0, 'depth <=0 '

    in_filter = filter_for_depth(depth+1)
    mid_filter = filter_for_depth(depth)
    out_filter = filter_for_depth(depth-1)

    layers = []
    layers.append(conv2x2(in_filter, mid_filter))
    layers.append(conv2x2(mid_filter, mid_filter))
    layers.append(upconv2x2(mid_filter, out_filter))

    return nn.Sequential(*layers)

def post_block(num_classes):
    """conv->conv->up"""
    in_filter = filter_for_depth(1)
    mid_filter = filter_for_depth(0)

    layers = []
    layers.append(conv2x2(in_filter, mid_filter))
    layers.append(conv2x2(mid_filter, mid_filter))
    layers.append(nn.Conv2d(mid_filter, num_classes, kernel_size=1))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, color_dim=3, num_classes=2):
        super(UNet, self).__init__()

        self.input = pre_block(color_dim)
        self.con1 = contraction(1)
        self.con2 = contraction(2)
        self.con3 = contraction(3)
        self.con4 = contraction(4)

        self.up4 = upsample(4)
        self.exp3 = expansion(3)
        self.exp2 = expansion(2)
        self.exp1 = expansion(1)
        self.output = post_block(num_classes)

    def forward(self, x):
        c0 = self.input(x)#(1L, 16L, 512L, 512L)
        c1 = self.con1(c0)#(1L, 32L, 256L, 256L)
        c2 = self.con2(c1)#(1L, 64L, 128L, 128L)
        c3 = self.con3(c2)#(1L, 128L, 64L, 64L)
        c4 = self.con4(c3)#(1L, 256L, 32L, 32L)

        u3 = self.up4(c4)#(1L, 128L, 64L, 64L)
        u3_c3 = torch.cat((u3, c3), 1)

        u2 = self.exp3(u3_c3)#(1L, 64L, 128L, 128L)
        u2_c2 = torch.cat((u2, c2), 1)

        u1 = self.exp2(u2_c2)
        u1_c1 = torch.cat((u1, c1), 1)

        u0 = self.exp1(u1_c1)
        u0_c0 = torch.cat((u0, c0), 1)

        """print 'c0 : ', c0.size()
        print 'c1 : ', c1.size()
        print 'c2 : ', c2.size()
        print 'c3 : ', c3.size()
        print 'c4 : ', c4.size()
        print 'u3 : ', u3.size()
        print 'u2 : ', u2.size()
        print 'u1 : ', u1.size()
        print 'u0 : ', u0.size()"""

        output = self.output(u0_c0)

        return output

if __name__ == "__main__":
    unet_2d = UNet(1, 2)

    x = torch.rand(1, 1, 320, 480)
    x = torch.autograd.Variable(x)

    print (x.size())

    y = unet_2d(x)

    print("-----------------------------")
    print(y.size())