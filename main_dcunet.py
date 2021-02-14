import torch
from torch.utils.data.dataset import random_split
from torchaudio import datasets

from dcunet.dcunet import DCUnet10
from dcunet.train import train, wsdr_fn
from utils.data import get_data, preprocess, WrappedDataLoader

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = datasets.VCTK_092(root="data", download=False)

ds_size = len(dataset)
train_i = int(0.1 * ds_size)
train_i += train_i % 2
val_i = int(0.11 * ds_size) - train_i
val_i += val_i % 2
test_i = int(0.12 * ds_size) - train_i - val_i
test_i += test_i % 2
other = ds_size - train_i - val_i - test_i

train_ds, val_ds, test_ds, other = random_split(dataset, lengths=[train_i, val_i, test_i, other],
                                                generator=torch.Generator().manual_seed(42))

batch_size = 1
SAMPLE_RATE = 48000
N_FFT = SAMPLE_RATE * 64 // 1000 + 4
HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4
zero_q = 0.9
zero_f = 0.2
one_q = 0.9
one_f = 0.5

train_dl, val_dl, test_dl = get_data(train_ds, val_ds, test_ds, batch_size)
train_dl = WrappedDataLoader(train_dl, preprocess, HOP_LENGTH, dev, zero_q=zero_q, zero_f=zero_f, one_q=one_q,
                             one_f=one_f)
val_dl = WrappedDataLoader(val_dl, preprocess, HOP_LENGTH, dev, zero_q=zero_q, zero_f=zero_f, one_q=one_q, one_f=one_f)
test_dl = WrappedDataLoader(test_dl, preprocess, HOP_LENGTH, dev, zero_q=zero_q, zero_f=zero_f, one_q=one_q,
                            one_f=one_f)

dcunet10 = DCUnet10(N_FFT, HOP_LENGTH).to(dev)

loss_fn = wsdr_fn
optimizer = torch.optim.Adam(dcunet10.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_losses, test_losses = train(dcunet10, train_dl, val_dl, loss_fn, optimizer, scheduler, 100, dev)
