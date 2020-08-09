import torch
import torch.distributed


class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self, device, distributed, topk=(1,5)):
        super(Metric, self).__init__()
        self._topk = topk
        self._distributed = distributed
        self._maxk = max(topk)

        self._n_images = torch.zeros(1, dtype=torch.int64, requires_grad=False).to(device)
        self._n_correct = torch.zeros(len(topk), dtype=torch.int64, requires_grad=False).to(device)
        self._accuracy = torch.zeros(len(topk), dtype=torch.float, requires_grad=False).to(device)

    def reset(self):
        self._n_images.fill_(0)
        self._n_correct.fill_(0)

    def update(self, predictions, labels):
        with torch.no_grad():
            n_images = torch.tensor(predictions.size(1), dtype=torch.int64, requires_grad=False).to(predictions.device)
            n_correct = torch.zeros(len(self._topk), dtype=torch.int64, requires_grad=False).to(predictions.device)

            correct = predictions.eq(labels.expand_as(predictions))

            for i, k in enumerate(self._topk):
                n_correct[i] += correct[:k].sum()

            if self._distributed:
                n_images = self.reducer(n_images)
                n_correct = self.reducer(n_correct)

            self._n_images += n_images
            self._n_correct += n_correct

            for i, k in enumerate(self._topk):
                self._accuracy[i] = self._n_correct[i].float() / self._n_images.float()

    def calculate(self):
        return self._accuracy.clone()

    @staticmethod
    def reducer(t):
        rt = t.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)

        return rt
