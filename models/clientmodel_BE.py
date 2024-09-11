from collections import OrderedDict
import torch
import torch.nn as nn



class UNET_BE(nn.Module):
    def __init__(self, out_channels=5, init_features=32):
        super(UNET_BE, self).__init__()
        features = init_features

        self.decoder1 = UNET_BE._block(features, features, name="dec1")  #in_channels = 32, out_channeks=32
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        dec1 = self.decoder1(x) #32 inchannels, #32 outchannels
        return self.conv(dec1) #32 inchannels, #5 outchannels

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )