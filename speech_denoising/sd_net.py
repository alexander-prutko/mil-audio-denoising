from torch import nn


class SpeechDenoising(nn.Module):
    def __init__(self):
        super().__init__()
        W = 64
        self.conv_layer_list = [None] * 16
        self.relu_list = [nn.LeakyReLU(0.2)] * 15
        self.batch_norm_list = [nn.BatchNorm1d(64)] * 15
        self.conv_layer_list[0] = nn.Conv1d(in_channels=1, out_channels=W, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        for i in range(13):
            self.conv_layer_list[i + 1] = nn.Conv1d(in_channels=W, out_channels=W, kernel_size=3, stride=1, padding=2 ** i, bias=True, dilation=2 ** i)
        self.conv_layer_list[13] = nn.Conv1d(in_channels=W, out_channels=W, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.conv_layer_list[14] = nn.Conv1d(in_channels=W, out_channels=W, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.conv_layer_list[15] = nn.Conv1d(in_channels=W, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        self.conv_module_list = nn.ModuleList(self.conv_layer_list)
        self.relu_module_list = nn.ModuleList(self.relu_list)
        self.batch_norm_module_list = nn.ModuleList(self.batch_norm_list)

    def forward(self, x):
        y = x
        for i in range(15):
            y = self.relu_list[i](self.batch_norm_list[i](self.conv_layer_list[i](y)))
        y = self.conv_layer_list[15](y)

        return y
