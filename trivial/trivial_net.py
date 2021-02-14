import torch
from torch import nn


class TrivialCADAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2048, stride=512, padding=1023, bias=True)
        self.deconv_1 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2048, stride=512, padding=1023,
                                           bias=True)

    def encoder(self, x):
        y = torch.tanh(self.conv_1(x))
        return y

    def decoder(self, y):
        x_hat = self.deconv_1(y)
        return x_hat

    def forward(self, x):
        y = self.encoder(x)
        x_hat = self.decoder(y)
        return x_hat