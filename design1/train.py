import torch
from torch import cuda, optim, autograd, nn
from my_utils import *
from nets import BigNet
from torch.utils.data import dataloader
from distill import change_lr, test, get_loaders, get_correct, big_loss_fn, small_loss_fn
import cv2
import numpy as np

MAX_EPOCH = 20
MAX_EPOCH_STEP2 = 40


class DiscriminatorNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(DiscriminatorNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        lst = []
        out = self.fc1(x.view(x.size(0), -1))
        lst.append(out)
        out = self.fc2(out)
        lst.append(out)
        return out, lst


class ConfidenceNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(ConfidenceNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Softmax(dim=1),
            nn.Linear(10, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc1(x)
        return self.fc2(out)


class ConfidenceNet2(nn.Module):
    def __init__(self, hidden_size=128):
        super(ConfidenceNet2, self).__init__()
        self.fc1 = nn.Sequential(
            # nn.Softmax(dim=1),
            nn.BatchNorm1d(128),
            nn.Linear(128, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc1(x)
        return self.fc2(out)


class SmallNetDesign1(nn.Module, IConfidence):
    def __init__(self, hidden_size=1000, con_hidden_size=512):
        super(SmallNetDesign1, self).__init__()
        self.D_net = DiscriminatorNet(hidden_size)
        # self.C_net = ConfidenceNet(con_hidden_size)
        self.C_net = ConfidenceNet2(con_hidden_size)
        self.lst = []

    def forward(self, x):
        out, self.lst = self.D_net(x)
        return out

    def confidence(self, x=None):
        if x is not None:
            with torch.no_grad():
                self.forward(x)
        out = self.C_net(self.lst[0])
        return out


def test_confidence(net: SmallNetDesign1, test_loader: dataloader.DataLoader, con_thresh=0.5):
    params = torch.zeros(8).cuda()
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            out = net(x)
            out_con = net.confidence()
            target = get_confidence_GT_from_logits(out, y)
            params += get_correct_with_confidence(out_con, target, con_thresh, as_tensor=True)
    # acc, confidence acc, acc within pred_right, acc within pred_wrong, upload_rate
    return params[0] / params[1], params[2] / params[3], params[4] / params[5], \
        params[6] / params[7], 1. - params[1].item() / len(test_loader.dataset)


def train_small_net(big_net: BigNet(), train_loader: dataloader.DataLoader,
                    test_loader: dataloader.DataLoader, skip_step1=False):
    print('training small net')
    model = SmallNetDesign1(hidden_size=128, con_hidden_size=512).cuda()
    start_lr = 1e-4
    # decay = 1e-6

    neg_weight = 6.
    con_thresh = 0.5

    def step1():
        optimizer = optim.Adam(model.parameters(), lr=start_lr)
        total = len(train_loader.dataset)
        for epoch in range(MAX_EPOCH):
            total_loss, correct = 0., 0
            for i, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                logits = model(x)
                with torch.no_grad():
                    big_logits = big_net(x)
                loss = small_loss_fn(logits, big_logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += get_correct(logits, y)
            acc = correct / total
            test_loss, test_acc = test(model, test_loader, big_loss_fn)
            print('Epoch %2d: train loss= %7.4f, acc= %.4f; test loss= %7.4f, acc= %.4f' % (
                epoch + 1, total_loss, acc, test_loss, test_acc
            ))

    if skip_step1:
        print('skip step 1')
        model.D_net.load_state_dict(torch.load('D_net.pth'))
    else:
        step1()
        torch.save(model.D_net.state_dict(), 'D_net.pth')

    def step2():
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        print('training confidence')
        for epoch in range(MAX_EPOCH_STEP2):
            total, con_total = 0, 0
            total_loss, correct, con_correct = 0., 0, 0
            for i, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                with torch.no_grad():
                    out = model(x)
                y_pred = model.confidence()
                target = get_confidence_GT_from_logits(out, y)
                loss = confidence_loss(y_pred, target, neg_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tp1, tp2, tp3, tp4, _ = get_correct_with_confidence(y_pred, target, con_thresh)

                total_loss += loss.item()
                correct += tp1
                total += tp2
                con_correct += tp3
                con_total += tp4
            acc = correct / total if total > 0 else 1.
            acc_con = con_correct / con_total if con_total > 0 else 1.
            print('Epoch %2d:\ntrain loss= %7.4f, acc= %.4f, con_acc= %.4f, upload= %.4f' % (
                epoch + 1, total_loss, acc, acc_con, 1. - total / len(train_loader.dataset)
            ))
            print('test acc= %.4f, confidence acc= %.4f, within right= %.4f, within wrong= %.4f, upload= %.4f'
                  % test_confidence(model, test_loader, con_thresh))
        pass

    step2()
    torch.save(model.C_net.state_dict(), 'C_net.pth')
    return model


def view(test_loader: dataloader.DataLoader, model: SmallNetDesign1, confidence_flag=True):
    clear_img_lst, vague_img_lst = [], []
    for i, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            D_out = model(x)
            C_out = model.confidence().view(-1)
            C_target = get_confidence_GT_from_logits(D_out, y)
        for j in range(C_target.size(0)):
            if not confidence_flag:
                if C_target[j] == 0 and len(vague_img_lst) < 256:
                    vague_img_lst.append(x[j].cpu())
                if C_target[j] == 1 and len(clear_img_lst) < 256:
                    clear_img_lst.append(x[j].cpu())
            else:
                if C_out[j] <= 0.5 and len(vague_img_lst) < 256:
                    vague_img_lst.append(x[j].cpu())
                if C_out[j] > 0.5 and len(clear_img_lst) < 256:
                    clear_img_lst.append(x[j].cpu())
        if len(vague_img_lst) == 256 and len(clear_img_lst) == 256:
            break

    def merge(_lst: list):
        _ret = np.zeros((28 * 16, 28 * 16), dtype=np.uint8)
        ix, iy = 0, 0
        for ii in range(len(_lst)):
            sx, sy = 28 * ix, 28 * iy
            _img = _lst[ii].view(28, 28).numpy()
            _img = (_img * 255.).astype(dtype=np.uint8)
            _ret[sx:sx + 28, sy:sy + 28] = _img

            iy += 1
            if iy == 16:
                ix += 1
                iy = 0
        return _ret
    clear_imgs = merge(clear_img_lst)
    vague_imgs = merge(vague_img_lst)
    cv2.imshow('clear imgs', clear_imgs)
    cv2.imshow('vague imgs', vague_imgs)
    cv2.waitKey(0)


def main():
    train_loader, test_loader = get_loaders(root='../data')
    big_net = BigNet().cuda()
    big_net.load_state_dict(torch.load('../big_net3.pth'))
    train_small_net(big_net, train_loader, test_loader, skip_step1=True)
    # model = SmallNetDesign1(hidden_size=128, con_hidden_size=512).cuda()
    # model.load_state_dict(torch.load('net_step2.pth'))
    # view(test_loader, model, confidence_flag=False)
    pass


if __name__ == '__main__':
    main()
    # train_ConNet()
    # build_con_dataset()
