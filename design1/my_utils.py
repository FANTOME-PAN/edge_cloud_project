import torch
from torch import cuda, optim, autograd, nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from nets import BigNet
from torch.utils.data import dataloader
from distill import change_lr, test, get_loaders, get_correct, big_loss_fn, small_loss_fn
import cv2


class IConfidence:
    def confidence(self, x: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError()


def confidence_loss(y_pred: torch.Tensor, y: torch.Tensor, neg_weight=10.):
    y = y.view(y_pred.size())
    assert y.size(1) == 1
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


def get_correct_with_confidence(out_con: torch.Tensor, y_con: torch.Tensor, con_thresh=0.5, as_tensor=False):
    con_over_thresh = out_con > con_thresh
    pred_right = y_con.type(torch.uint8).view(-1, 1)
    assert pred_right.shape == con_over_thresh.shape
    correct = (pred_right * con_over_thresh).sum().item()
    total = con_over_thresh.sum().item()

    con = out_con > 0.5
    tmp = con == pred_right
    con_correct = tmp.sum().item()
    con_total = con.size(0)
    correct_1 = (tmp * pred_right).sum().item()
    correct_2 = con_correct - correct_1
    total_1 = pred_right.sum().item()
    total_2 = con_total - total_1
    details = (correct_1, total_1, correct_2, total_2)
    # correct and total within confidence; number of correct confidence and total confident case.
    if as_tensor:
        ret = torch.tensor([correct, total, con_correct, con_total, *details], dtype=torch.float).cuda()
        return ret
    return correct, total, con_correct, con_total, details


def get_confidence_GT(y_pred: torch.Tensor, y_raw: torch.Tensor):
    assert len(y_raw.shape) == 1
    return y_pred.eq(y_raw)


def get_confidence_GT_from_logits(logits: torch.Tensor, y_raw: torch.Tensor):
    assert len(y_raw.shape) == 1
    return logits.max(1)[1].eq(y_raw)

