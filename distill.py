import torch
from torch import cuda, optim, autograd, nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import dataloader
from nets import *


MAX_EPOCH_BIG = 40


def get_correct(logits, y):
    return logits.max(1)[1].eq(y).sum().item()


def get_loaders(train_batch_size=128, test_batch_size=1024, shuffle=False, root='./data'):

    train_set = MNIST(
        root=root,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_loader = dataloader.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True
    )

    test_set = MNIST(
        root=root,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    test_loader = dataloader.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test_loader


def test(net, test_loader: dataloader.DataLoader, loss_fn):
    total = len(test_loader.dataset)
    correct = 0
    total_loss = 0.
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            logits = net(x)
            total_loss += loss_fn(logits, y).item()
            correct += get_correct(logits, y)
    acc = correct / total
    return total_loss, acc


def change_lr(optimizer: optim.Optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr


def train_big_net(train_loader: dataloader.DataLoader, test_loader: dataloader.DataLoader, path=None, start_lr=1e-4):
    print('training big net')
    model = BigNet().cuda()
    if path is not None:
        model.load_state_dict(torch.load(path))
    loss_fn = big_loss_fn
    # start_lr = start_lr
    decay = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    total = len(train_loader.dataset)
    cnt = 0
    for epoch in range(MAX_EPOCH_BIG):
        total_loss, correct = 0., 0
        for i, (x, y) in enumerate(train_loader):

            x = x.cuda()
            y = y.cuda()
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1
            change_lr(optimizer, start_lr / (1. + cnt * decay))
            total_loss += loss.item()
            correct += get_correct(logits, y)
            # print('step %d' % cnt)
        acc = correct / total
        test_loss, test_acc = test(model, test_loader, loss_fn)
        print('Epoch %2d: train loss= %7.4f, acc= %.4f; test loss= %7.4f, acc= %.4f' % (
            epoch + 1, total_loss, acc, test_loss, test_acc
        ))
    torch.save(model.state_dict(), 'big_net3.pth')
    return model


def train_small_net(big_net: BigNet, train_loader: dataloader.DataLoader, test_loader: dataloader.DataLoader):
    print('training small net')
    model = SmallNet().cuda()
    loss_fn = small_loss_fn
    start_lr = 1e-4
    # decay = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    total = len(train_loader.dataset)
    cnt = 0
    for epoch in range(MAX_EPOCH_BIG):
        total_loss, correct = 0., 0
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            logits = model(x)
            with torch.no_grad():
                big_logits = big_net.forward(x)
            loss = loss_fn(logits, big_logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cnt += 1
            # change_lr(optimizer, start_lr / (1. + cnt * decay))
            total_loss += loss.item()
            correct += get_correct(logits, y)
        acc = correct / total
        test_loss, test_acc = test(model, test_loader, big_loss_fn)
        print('Epoch %2d: train loss= %7.4f, acc= %.4f; test loss= %7.4f, acc= %.4f' % (
            epoch + 1, total_loss, acc, test_loss, test_acc
        ))
    torch.save(model.state_dict(), 'small_net.pth')
    return model


def main():
    train_loader, test_loader = get_loaders()
    big_net = train_big_net(train_loader, test_loader, path='big_net2.pth', start_lr=5e-5)
    # big_net = BigNet().cuda()
    # big_net.load_state_dict(torch.load('big_net2.pth'))
    # test_loss, test_acc = test(big_net, test_loader, big_loss_fn)
    # print(test_acc)
    # small_net = train_small_net(big_net, train_loader, test_loader)
    pass


if __name__ == '__main__':
    main()


