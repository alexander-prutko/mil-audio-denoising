import torch
from torch import optim

from trivial.trivial_net import TrivialCADAE


def loss_batch(model, loss_func, x, y, opt=None):
    x_hat = model(x)
    loss = loss_func(x_hat, y.narrow(2, 0, x_hat.shape[2]))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), x.shape[0]


def fit(epochs, model, loss_func, opt, train_dl, val_dl):
    train_size = len(train_dl)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_dl):
            bs = x.shape[0]
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            _loss, _len = loss_batch(model, loss_func, x, y, opt)

            if batch_idx % 1 == 0:
                line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
                    epoch, batch_idx * bs, train_size*bs, 100. * batch_idx / train_size)
                losses = '{}: {:.10f}'.format("Trivial", _loss / _len)
                print(line + losses)

        batch_idx += 1
        line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
            epoch, min(batch_idx * bs, train_size), train_size*bs, 100. * batch_idx / train_size)
        losses = '{}: {:.10f}'.format("Trivial", _loss / _len)
        print(line + losses)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, x.unsqueeze(1), y.unsqueeze(1)) for (x, y) in val_dl]
            )
        val_loss = sum(losses) / sum(nums)

        print("Validation Epoch: {}\tLosses {}: {:.10f}".format("Trivial", epoch, val_loss))


def get_model(dev):
    net = TrivialCADAE()
    net.to(dev)
    return net, optim.Adam(net.parameters())