import torch
from torch import nn


class TrivialCADAE2(nn.Module):
    def __init__(self):
        super().__init__()
        W = 64
        self.convs = [None] * 16
        self.convs[0] = nn.Conv1d(in_channels=1, out_channels=W, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        for i in range(13):
            self.convs[i + 1] = nn.Conv1d(in_channels=W, out_channels=W, kernel_size=3, stride=1, padding=2**i, bias=True, dilation=2**i)
        self.convs[13] = nn.Conv1d(in_channels=W, out_channels=W, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.convs[14] = nn.Conv1d(in_channels=W, out_channels=W, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.convs[15] = nn.Conv1d(in_channels=15*W, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)

    def forward(self, x):
        outs = [None] * 15
        for i in range(15):
            outs[i] = self.convs[i](x)
        y = torch.cat(outs, 1)
        return y