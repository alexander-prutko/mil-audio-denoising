import torch
from torch.utils.data.dataset import random_split
from torchaudio import datasets

from dcunet.dcunet import DCUnet10
from utils.data import get_data, preprocess, WrappedDataLoader
from utils.train import train, wsdr_fn

print(torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = datasets.VCTK_092(root="data", download=False)

ds_size = len(dataset)
train_i = int(0.01 * ds_size)
train_i += train_i % 2
val_i = int(0.011 * ds_size) - train_i
val_i += val_i % 2
test_i = ds_size - train_i - val_i

train_ds, val_ds, test_ds = random_split(dataset, lengths=[train_i, val_i, test_i],
                                         generator=torch.Generator().manual_seed(42))

batch_size = 1
DEVICE = dev
SAMPLE_RATE = 48000
N_FFT = SAMPLE_RATE * 64 // 1000 + 4
HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4

train_dl, val_dl, test_dl = get_data(train_ds, val_ds, test_ds, batch_size)
train_dl = WrappedDataLoader(train_dl, preprocess, HOP_LENGTH, DEVICE)
val_dl = WrappedDataLoader(val_dl, preprocess, HOP_LENGTH, DEVICE)
test_dl = WrappedDataLoader(test_dl, preprocess, HOP_LENGTH, DEVICE)

dcunet10 = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)

loss_fn = wsdr_fn
optimizer = torch.optim.Adam(dcunet10.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_losses, test_losses = train(dcunet10, train_dl, val_dl, loss_fn, optimizer, scheduler, 100, dev)
