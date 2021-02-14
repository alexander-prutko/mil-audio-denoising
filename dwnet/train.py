import numpy as np
import torch


def loss_batch(model, loss_func, x, y, opt=None):
    x_hat = model(x).unsqueeze(1)
    loss = loss_func(x_hat, y.narrow(2, 0, x_hat.shape[2]))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), x.shape[0]


def fit(epochs, model, loss_func, opt, train_dl, val_dl, dev):
    sampleSize = 2048 * 2
    train_size = len(train_dl)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_dl):
            bs = x.shape[0]
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            startx = 0
            idx = np.arange(startx, x.shape[-1], sampleSize)
            _loss, _len = 0, 0
            for i, ind in enumerate(idx):
                if (x[0, 0, ind:ind + sampleSize].shape[0] < (sampleSize)): break
                data = x[:, :, ind:ind + sampleSize].to(dev)
                target = y[:, :, ind:ind + sampleSize].to(dev)
                _loss_, _len_ = loss_batch(model, loss_func, data, target, opt)
                _len += _len_
                _loss += _loss_

            if batch_idx % 1 == 0:
                line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
                    epoch, batch_idx * bs, train_size, 100. * batch_idx / train_size)
                losses = '{}: {:.10f}'.format("Trivial", _loss / _len)
                print(line + losses)

        #     else:
        batch_idx += 1
        line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
            epoch, min(batch_idx * bs, train_size), train_size, 100. * batch_idx / train_size)
        losses = '{}: {:.10f}'.format("Trivial", _loss / _len)
        print(line + losses)

        model.eval()
        with torch.no_grad():
            for (x, y) in val_dl:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                startx = 0
                idx = np.arange(startx, x.shape[-1], sampleSize)
                _loss, _len = 0, 0
                for i, ind in enumerate(idx):
                    if (x[0, 0, ind:ind + sampleSize].shape[0] < (sampleSize)): break
                    data = x[:, :, ind:ind + sampleSize].to(dev)
                    target = y[:, :, ind:ind + sampleSize].to(dev)
                    _loss_, _len_ = loss_batch(model, loss_func, data, target)
                    _len += _len_
                    _loss += _loss_
        val_loss = _loss / _len

        print("Validation Epoch: {}\tLosses {}: {:.10f}".format("Trivial", epoch, val_loss))