import torch
from torch import cuda, optim, autograd, nn
import torch.nn.functional as F


class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()

        def block(i_chnl, o_chnl):
            return [
                nn.Conv2d(i_chnl, o_chnl, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.2)
            ]
        self.features = nn.Sequential(
            *block(1, 32),
            *block(32, 64),
            *block(64, 128)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 625),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(625, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def big_loss_fn(y_pred, y):
    return F.cross_entropy(y_pred, y)


def small_loss_fn(y_pred, y_pred_big_net, y, temperature=1.):
    y1 = F.log_softmax(y_pred / temperature, dim=1)
    y2 = y_pred
    y_pred_big_net = F.softmax(y_pred_big_net, dim=1)
    loss1 = F.cross_entropy(y2, y)
    loss2 = -(y_pred_big_net * y1).sum(dim=1).mean()
    return loss1 + loss2
