import gc

import torch
from tqdm import tqdm


def train_epoch(net, train_loader, loss_fn, optimizer, dev):
    net.train()
    train_ep_loss = 0.
    counter = 0
    for batch_idx, (noisy_x, clean_x) in enumerate(train_loader):
        batch_size = noisy_x.shape[0]
        if batch_idx % 100 == 99:
            print(batch_idx, "/", len(train_loader) // batch_size, train_ep_loss / counter)

        noisy_x, clean_x = noisy_x.to(dev), clean_x.to(dev)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


def test_epoch(net, test_loader, loss_fn, dev):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    for noisy_x, clean_x in test_loader:

        # get the output from the model
        noisy_x, clean_x = noisy_x.to(dev), clean_x.to(dev)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        test_ep_loss += loss.item()

        counter += 1

    test_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss


def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs, dev):
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        # first evaluating for comparison
        if e == 0:
            with torch.no_grad():
                test_loss = test_epoch(net, test_loader, loss_fn, dev)

            test_losses.append(test_loss)
            print("Loss before training:{:.6f}".format(test_loss))

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer, dev)
        scheduler.step()
        with torch.no_grad():
            test_loss = test_epoch(net, test_loader, loss_fn, dev)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Loss: {:.6f}...".format(train_loss),
              "Test Loss: {:.6f}".format(test_loss))
    return train_losses, test_losses


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