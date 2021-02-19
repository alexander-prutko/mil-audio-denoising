import torch
from torch import optim

from speech_denoising.sd_net import SpeechDenoising


def loss_batch(model, loss_func, x, y, opt=None):
    x_hat = model(x)
    loss = loss_func(x, x_hat, y.narrow(2, 0, x_hat.shape[2]))

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
    net = SpeechDenoising()
    net.to(dev)
    return net, optim.Adam(net.parameters())

def wsdr_fn(x, y_pred_, y_true, eps=1e-8):
    # to time-domain waveform
    y_true = torch.squeeze(y_true, 1)
    x = torch.squeeze(x, 1)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    if y_pred.shape[1] < y_true.shape[1]:
        y_true = y_true.narrow(1, 0, y_pred.shape[1])
        x = x.narrow(1, 0, y_pred.shape[1])
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true ** 2, dim=1) / (torch.sum(y_true ** 2, dim=1) + torch.sum(z_true ** 2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

def wsdr_on_noisy_data(data_loader, dev):
    test_ep_loss = 0.
    counter = 0.
    for i, (noisy_x, clean_x) in enumerate(data_loader):
        noisy_x, clean_x = noisy_x.to(dev), clean_x.to(dev)
        loss = wsdr_fn(noisy_x, noisy_x, clean_x)
        test_ep_loss += loss.item()

        counter += 1
    test_ep_loss /= counter
    return test_ep_loss


def wsdr_on_processed_data(data_loader, dev, net):
    net.to(dev)
    with torch.no_grad():
        test_ep_loss = 0.
        counter = 0.
        for i, (noisy_x, clean_x) in enumerate(data_loader):
            noisy_x, clean_x = noisy_x.to(dev), clean_x.to(dev)
            pred_x = net(noisy_x.unsqueeze(1))
            loss = wsdr_fn(noisy_x, pred_x, clean_x)
            test_ep_loss += loss.item()

            counter += 1
        test_ep_loss /= counter
    return test_ep_loss
