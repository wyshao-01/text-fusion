import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import utils
import math
from Vit import Spatial
from Vit import Channel
from Spatial_Vit import Cross_Spatial
from Channel_Vit import Cross_Channel


# Convolution operation 填充卷积激活
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Prelu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Prelu = use_Prelu
        self.LeakyReLU=nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Prelu is True:
            out = self.LeakyReLU(out)
        return out

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                  dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.LeakyReLU = nn.LeakyReLU(0.1)
    def forward(self, x):
        x1=self.reflection_pad(x)
        out_normal = self.conv(x1)
        out_normal = self.LeakyReLU(out_normal)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        kernel_size_2 = 3
        self.save = utils.save_feat
        # encoder_convlayer
        self.SFB_ir = nn.Sequential(ConvLayer(1,32,3,1,True),ConvLayer(32, 32, 3, 1, True))
        self.SFB_vis = nn.Sequential(ConvLayer(1, 32, 3, 1, True), ConvLayer(32, 32, 3, 1, True))

        self.c_vit = Channel(embed_dim=128, patch_size=16, channel=32)  #128  16
        self.s_vit = Spatial(embed_dim=64,patch_size=8,channel=32)      #64   8
        self.cro_cha_vit = Cross_Channel(embed_dim=128,patch_size=16,channel=32)
        self.cro_spa_vit = Cross_Spatial(embed_dim=64,patch_size=8,channel=32)

        self.conv_1 = ConvLayer(64, 32, 3, stride=1, use_Prelu=True)
        self.conv_2 = ConvLayer(32, 16, 3, stride=1, use_Prelu=True)
        self.conv_3 = ConvLayer(16, 1, 1, stride=1, use_Prelu=True)
    def forward(self,input_ir,input_vis):
    #Encoder
        ir1 = self.SFB_ir(input_ir)# 16    256

        vis1 = self.SFB_vis(input_vis)

        cross_ir = self.cro_cha_vit(ir1, vis1) + ir1
        cross_vis = self.cro_cha_vit(vis1, ir1) + vis1

        cross_text_ir = self.s_vit(cross_ir) + cross_ir
        cross_text_vis = self.s_vit(cross_vis) + cross_vis

        fuse = torch.cat([cross_text_ir,cross_text_vis],dim=1)

        final_out = self.conv_3(self.conv_2(self.conv_1(fuse)))

        return final_out



