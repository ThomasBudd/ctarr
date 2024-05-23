import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def DoubleConvNormNonlin(in_channels,
                         out_channels,
                         first_stride):
    
    conv1 = nn.Conv3d(in_channels, 
                      out_channels, 
                      kernel_size=3,
                      padding=1,
                      stride=first_stride,
                      bias=False)    
    norm1 = nn.InstanceNorm3d(out_channels, affine=True)
    nonlin1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    conv2 = nn.Conv3d(out_channels, 
                      out_channels, 
                      kernel_size=3,
                      padding=1,
                      bias=False)    
    norm2 = nn.InstanceNorm3d(out_channels, affine=True)
    nonlin2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    return nn.Sequential(conv1, norm1, nonlin1, conv2, norm2, nonlin2)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=20):
        super().__init__()
        
        filters_list = [in_channels, 32, 64, 128, 256]

        # first all blocks of the encoder
        self.encoder_modules = []
        for i in range(4):
            self.encoder_modules.append(DoubleConvNormNonlin(filters_list[i],
                                                             filters_list[i+1],
                                                             1 if i == 0 else 2))
        
        self.encoder_modules = nn.ModuleList(self.encoder_modules)

        # now blocks of the decoder
        self.decoder_modules = []
        for i in range(3):
            self.decoder_modules.append(DoubleConvNormNonlin(2*filters_list[i+1],
                                                             filters_list[i+1],
                                                             1))
        
        self.decoder_modules = nn.ModuleList(self.decoder_modules)
        
        # now all the upsamplings
        self.upsampling_modules = []
        for i in range(3):
            upsampling = nn.ConvTranspose3d(filters_list[i+2], filters_list[i+1], 2, stride=2, bias=False)
            self.upsampling_modules.append(upsampling)
        
        self.upsampling_modules = nn.ModuleList(self.upsampling_modules)
        
        # now all the Logits
        self.logit_modules = []
        for i in range(2):
            logit = nn.Conv3d(in_channels=filters_list[i+1],
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False)
            self.logit_modules.append(logit)
        # the second logits were only used for training and are included only
        # to prevent errors when loading weights
        self.logit_modules = nn.ModuleList(self.logit_modules)

    def forward(self, xb):
        
        # keep all out tensors from the contracting path
        skip_list = []
        
        # encoder
        for i in range(3):
            xb = self.encoder_modules[i](xb)
            skip_list.append(xb)
        # bottom block, no need for storing the skip tensor
        xb = self.encoder_modules[3](xb)

        # decoder
        for i in range(2,-1,-1):
            xb = self.upsampling_modules[i](xb)
            xb = torch.cat([xb, skip_list[i]], 1)
            del skip_list[i]
            xb = self.decoder_modules[i](xb)

        return self.logit_modules[0](xb)
