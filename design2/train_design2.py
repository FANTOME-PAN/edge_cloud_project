import torch
from torch import cuda, optim, autograd, nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from nets import BigNet
from torch.utils.data import dataloader
from distill import change_lr, test, get_loaders, get_correct, big_loss_fn, small_loss_fn

MAX_EPOCH = 10
MAX_EPOCH_STEP2 = 10


class SmallNetDesign2(nn.Module):
    def __init__(self, hidden_size=1000):
        super(SmallNetDesign2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(hidden_size, 10)
        # self.con_fc = nn.Sequential(
        #     nn.Linear(hidden_size + 10, 1),
        #     nn.Sigmoid()
        # )
        self.con_fc1 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
        self.con_fc2 = nn.Sequential(
            nn.Linear(10, 1),
            nn.ReLU()
        )
        self.con_fc3 = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.con_in = [None, None]

    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1))
        self.con_in[0] = out.detach()
        out = self.fc2(out)
        self.con_in[1] = out.detach()
        return out

    def confidence(self, x=None):
        if x is not None:
            with torch.no_grad():
                self.forward(x)
        o1 = self.con_fc1(self.con_in[0])
        o2 = self.con_fc2(self.con_in[1])
        ret = self.con_fc3(torch.cat((o1, o2), dim=1))
        return ret


def confidence_loss(y_pred: torch.Tensor, y: torch.Tensor, neg_weight=10.):
    if y.dtype != torch.float:
        y = y.type(torch.float)
    # y_inverse[i] = not y[i]
    y_inverse = -y + 1.
    # if y[i] == 1 then y_k[i] == 1. else -1.
    y_k = y - y_inverse
    # if y[i] == 1 then y_neg[i] == 1. else neg_weight
    y_neg = y + neg_weight * y_inverse
    ret = -torch.log(y_pred * y_k + y_inverse) * y_neg
    return ret.mean()


def get_correct_with_confidence(out_con: torch.Tensor, y_con: torch.Tensor, con_thresh=0.5):
    con_over_thresh = out_con > con_thresh
    pred_right = y_con.view(-1, 1)
    correct = (pred_right * con_over_thresh).sum().item()
    total = con_over_thresh.sum().item()

    con = out_con > 0.5
    con_correct = (con == pred_right).sum().item()
    con_total = con.sum().item()

    # correct and total within confidence; number of correct confidence and total confident case.
    return correct, total, con_correct, con_total


def test_confidence(net: SmallNetDesign2, test_loader: dataloader.DataLoader, con_thresh=0.5):
    total = 0
    correct = 0
    con_correct = 0
    con_total = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            out = net(x)
            out_con = net.confidence()
            y_con = out.max(1)[1] == y
            tp1, tp2, tp3, tp4 = get_correct_with_confidence(out_con, y_con, con_thresh)
            correct += tp1
            total += tp2
            con_correct += tp3
            con_total += tp4
    acc = correct / total if total > 0 else 1.
    con_acc = con_correct / con_total if con_total > 0 else 1.
    return acc, con_acc


def train_small_net(big_net: BigNet(), train_loader: dataloader.DataLoader,
                    test_loader: dataloader.DataLoader, skip_step1=False):
    print('training small net')
    model = SmallNetDesign2().cuda()
    start_lr = 1e-4
    # decay = 1e-6

    neg_weight = 5.
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
        model.load_state_dict(torch.load('net_step1.pth'))
    else:
        step1()
        torch.save(model.state_dict(), 'net_step1.pth')

    def step2():
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
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
                target = out.max(1)[1] == y
                loss = confidence_loss(y_pred, target, neg_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tp1, tp2, tp3, tp4 = get_correct_with_confidence(y_pred, target, con_thresh)

                total_loss += loss.item()
                correct += tp1
                total += tp2
                con_correct += tp3
                con_total += tp4
            acc = correct / total if total > 0 else 1.
            acc_con = con_correct / con_total if con_total > 0 else 1.
            test_acc, test_acc_con = test_confidence(model, test_loader, con_thresh)
            print('Epoch %2d: train loss= %7.4f, acc= %.4f, con_acc= %.4f, con_rate= %.4f; '
                  'test acc= %7.4f, con_acc= %.4f' % (
                      epoch + 1, total_loss, acc, acc_con, total / len(train_loader.dataset), test_acc, test_acc_con
                  ))
        pass

    step2()
    torch.save(model.state_dict(), 'net_step2.pth')
    return model


def main():
    train_loader, test_loader = get_loaders(root='../data')
    big_net = BigNet().cuda()
    big_net.load_state_dict(torch.load('../big_net3.pth'))
    train_small_net(big_net, train_loader, test_loader, skip_step1=True)
    pass


if __name__ == '__main__':
    main()
