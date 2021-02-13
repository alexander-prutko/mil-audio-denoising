import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np


def pad_collate(batch):
    (xx, _, _, _, _) = zip(*batch)
    xx = [x.squeeze() for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).unsqueeze(1)

    return xx_pad


def get_data(train_ds, val_ds, test_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, collate_fn=pad_collate, shuffle=True),
        DataLoader(val_ds, batch_size=2 * bs, collate_fn=pad_collate),
        DataLoader(test_ds, batch_size=2 * bs, collate_fn=pad_collate),
    )


def preprocess(x, hop_length, dev):
    ns = int(((((x.shape[1] / hop_length - 4) / 2 - 3) / 2 - 1) / 2 - 1) / 2)
    ns = int(((((ns * 2 + 1) * 2 + 1) * 2 + 3) * 2 + 4) * hop_length)
    x = x.narrow(1, 0, ns)
    return x.to(dev)


class WrappedDataLoader:
    def __init__(self, dl, func, hop_length, dev):
        self.dl = dl
        self.func = func
        self.hop_length = hop_length
        self.dev = dev

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            x = self.make_audio_noise(b)
            x_noisy, y_true = self.func(x, self.hop_length, self.dev), \
                              self.func(b.squeeze(1), self.hop_length, self.dev)
            yield (x_noisy, y_true)

    def make_audio_noise(self, b):
        inter = torch.stft(b.squeeze(), n_fft=2048, return_complex=True)
        if len(inter.shape) == 2:
            inter = inter.unsqueeze(0)
        self.zero_random_indicies(inter)
        self.factor_random_indicies(inter)
        x = torch.istft(inter, n_fft=2048)
        return x

    def zero_random_indicies(self, inter):
        inter_mag = torch.abs(inter)
        inds = torch.where(inter_mag > torch.quantile(inter_mag, 0.9))
        ri = np.random.choice(inds[0].shape[0], inds[0].shape[0] // 5, replace=False)
        indicies = (inds[0][ri], inds[1][ri], inds[2][ri])
        inter[indicies] = 0

    def factor_random_indicies(self, inter):
        inter_mag = torch.abs(inter)
        inds = torch.where(inter_mag < torch.quantile(inter_mag, 0.9))
        ri = np.random.choice(inds[0].shape[0], inds[0].shape[0] // 2, replace=False)
        indicies = (inds[0][ri], inds[1][ri], inds[2][ri])
        inter[indicies] += 1
        inter[indicies] = inter[indicies] / torch.abs(inter[indicies])
