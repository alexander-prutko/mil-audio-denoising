import json

import torch
from torch import nn, optim
from torch.utils.data.dataset import random_split
from torchaudio import datasets

from dwnet.dwnet import DenoisingWavenet
from dwnet.train import fit
from utils.data import get_data, WrappedDataLoader, preprocess

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = datasets.VCTK_092(root="data", download=False)

ds_size = len(dataset)
train_i = int(0.001 * ds_size)
val_i = int(0.002 * ds_size) - train_i
test_i = int(0.003 * ds_size) - train_i - val_i
other = ds_size - train_i - val_i - test_i

train_ds, val_ds, test_ds, other = random_split(dataset, lengths=[train_i, val_i, test_i, other],
                                                generator=torch.Generator().manual_seed(42))

SAMPLE_RATE = 48000
N_FFT = SAMPLE_RATE * 64 // 1000 + 4
HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4
zero_q = 0.9
zero_f = 0.2
one_q = 0.9
one_f = 0.5
bs = 16

train_dl, val_dl, test_dl = get_data(train_ds, val_ds, test_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess, HOP_LENGTH, dev, zero_q=zero_q, zero_f=zero_f, one_q=one_q,
                             one_f=one_f)
val_dl = WrappedDataLoader(val_dl, preprocess, HOP_LENGTH, dev, zero_q=zero_q, zero_f=zero_f, one_q=one_q, one_f=one_f)
test_dl = WrappedDataLoader(test_dl, preprocess, HOP_LENGTH, dev, zero_q=zero_q, zero_f=zero_f, one_q=one_q,
                            one_f=one_f)

config = json.load(open("dwnet/config.json"))

model = DenoisingWavenet(config).to(dev)
opt = optim.Adam(model.parameters())
loss_func = nn.L1Loss()

fit(100, model, loss_func, opt, train_dl, val_dl, dev)
