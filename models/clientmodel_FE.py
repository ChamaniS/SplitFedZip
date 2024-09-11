from collections import OrderedDict
import torch
import torch.nn as nn


class UNET_FE(nn.Module):
    def __init__(self, in_channels=3, init_features=32):
        super(UNET_FE, self).__init__()
        features = init_features
        self.encoder1 = UNET_FE._block(in_channels, features, name="enc1")

    def forward(self, x):
        enc1 = self.encoder1(x) #3 inchannels, #32 outchannels
        return enc1

    @staticmethod
    def _block(in_channels, features, name):
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
                ]
            )
        )