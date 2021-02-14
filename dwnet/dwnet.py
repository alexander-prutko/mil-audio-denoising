import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dwnet import layers, util


class DenoisingWavenet(nn.Module):

    def __init__(self, config, input_length=None, target_field_length=None):
        super().__init__()

        self.config = config
        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']

        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)

        if input_length is not None:
            self.input_length = int(input_length)
            self.target_field_length = int(self.input_length - (self.receptive_field_length - 1))
        if target_field_length is not None:
            self.target_field_length = int(target_field_length)
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))
        else:
            self.target_field_length = int(config['model']['target_field_length'])
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))

        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        self.half_target_field_length = int(self.target_field_length / 2)
        self.half_receptive_field_length = int(self.receptive_field_length / 2)
        self.num_residual_blocks = len(self.dilations) * self.num_stacks

        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = self.input_length
        self.config['model']['target_field_length'] = self.target_field_length

        # Layers in the model
        self.conv1 = nn.Conv1d(1, self.config['model']['filters']['depths']['res'],
                               self.config['model']['filters']['lengths']['res'], stride=1, bias=False, padding=1)

        self.conv2 = nn.Conv1d(self.config['model']['filters']['depths']['res'],
                               self.config['model']['filters']['depths']['final'][0],
                               self.config['model']['filters']['lengths']['final'][0], stride=1, bias=False, padding=1)

        self.conv3 = nn.Conv1d(self.config['model']['filters']['depths']['final'][0],
                               self.config['model']['filters']['depths']['final'][1],
                               self.config['model']['filters']['lengths']['final'][1], stride=1, bias=False, padding=1)

        self.conv4 = nn.Conv1d(self.config['model']['filters']['depths']['final'][1], 1, 1, stride=1, bias=False,
                               padding=0)

        self.dilated_layers = nn.ModuleList(
            [dilated_residual_block(dilation, self.input_length, self.padded_target_field_length, self.config) for
             dilation in self.dilations])

    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_condition_input_length(self, representation):

        if representation == 'binary':
            return int(np.ceil(np.log2(self.num_condition_classes)))
        else:
            return self.num_condition_classes

    def forward(self, x):

        data_input = x  # ['data_input']

        data_expanded = data_input  # layers.expand_dims(data_input, 1)

        data_out = self.conv1(data_expanded)

        skip_connections = []
        for _ in range(self.num_stacks):
            for layer in self.dilated_layers:
                data_out, skip_out = layer(data_out)
                if skip_out is not None:
                    skip_connections.append(skip_out)

        data_out = torch.stack(skip_connections, dim=0).sum(dim=0)
        data_out = F.relu(data_out)

        data_out = self.conv2(data_out)

        data_out = F.relu(data_out)

        data_out = self.conv3(data_out)

        data_out = self.conv4(data_out)

        data_out_speech = data_out
        data_out_noise = data_input - data_out_speech

        data_out_speech = data_out_speech.squeeze_(1)

        data_out_noise = data_out_noise.squeeze_(1)

        return data_out_speech


class dilated_residual_block(nn.Module):

    def __init__(self, dilation, input_length, padded_target_field_length, config):
        super().__init__()
        self.dilation = dilation
        self.input_length = input_length
        #         self.condition_input_length = condition_input_length
        #         self.samples_of_interest_indices = samples_of_interest_indices
        self.padded_target_field_length = padded_target_field_length
        self.config = config
        self.conv1 = nn.Conv1d(self.config['model']['filters']['depths']['res'],
                               2 * self.config['model']['filters']['depths']['res'],
                               kernel_size=self.config['model']['filters']['lengths']['res'], stride=1, bias=False,
                               dilation=self.dilation,
                               padding=int(self.dilation))
        self.conv2 = nn.Conv1d(self.config['model']['filters']['depths']['res'],
                               self.config['model']['filters']['depths']['res'] +
                               self.config['model']['filters']['depths']['skip'],
                               1, stride=1, bias=False, padding=0)

    def forward(self, data_x):
        original_x = data_x

        # Data sub-block
        data_out = self.conv1(data_x)

        data_out_1 = layers.slicing(data_out, slice(0, self.config['model']['filters']['depths']['res'], 1), 1)

        data_out_2 = layers.slicing(data_out, slice(self.config['model']['filters']['depths']['res'],
                                                    2 * self.config['model']['filters']['depths']['res'], 1), 1)

        data_out_1 = data_out_1

        data_out_2 = data_out_2

        tanh_out = torch.tanh(data_out_1)
        sigm_out = torch.sigmoid(data_out_2)

        data_x = tanh_out * sigm_out

        data_x = self.conv2(data_x)

        res_x = layers.slicing(data_x, slice(0, self.config['model']['filters']['depths']['res'], 1), 1)

        skip_x = layers.slicing(data_x, slice(self.config['model']['filters']['depths']['res'],
                                              self.config['model']['filters']['depths']['res'] +
                                              self.config['model']['filters']['depths']['skip'], 1), 1)

        res_x = res_x + original_x

        return res_x, skip_x
