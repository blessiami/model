import torch
import torch.nn as nn


class SmoothCrossEntropy(nn.Module):
    def __init__(self, k=1000, eps=0.1, size_average=True):
        super(SmoothCrossEntropy, self).__init__()
        self._k = k
        self._eps = eps

        self._func = torch.mean if size_average else torch.sum
        self._logsoftmax = nn.LogSoftmax()

    def forward(self, predictions, labels):
        smooth_labels = torch.ones_like(predictions) * self._eps / (self._k - 1)
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self._eps)

        return self._func(torch.sum(-smooth_labels * self._logsoftmax(predictions), dim=1))
