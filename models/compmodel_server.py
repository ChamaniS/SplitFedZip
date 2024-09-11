import torch
import torch.nn as nn

from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

class COMP_NW_SERVER(CompressionModel):
    def __init__(self, N=64):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N))

    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods