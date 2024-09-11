from collections import OrderedDict
import torch
import torch.nn as nn



class UNET_BE(nn.Module):
    def __init__(self, out_channels=5, init_features=32):
        super(UNET_BE, self).__init__()
        features = init_features

        #self.conv_add = nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3,padding=1,bias=False,)
        self.decoder2_2 = UNET_BE._block0(features*2 , features * 2, name="dec2_2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features * 1, kernel_size=2, stride=2
        )
        self.decoder1 = UNET_BE._block2(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )


    def forward(self,enc1, x):
        dec1 = self.decoder2_2(x)
        dec1 = self.upconv1(dec1)
        dec1 = torch.cat((dec1, enc1), dim=1) #64
        dec1 = self.decoder1(dec1)     #64 inchannels, #32 outchannels
        return self.conv(dec1)




    @staticmethod
    def _block0(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv3",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm3", nn.BatchNorm2d(num_features=features)),
                    (name + "relu3", nn.ReLU(inplace=True)),
                ]
            )
        )

    def _block2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
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
