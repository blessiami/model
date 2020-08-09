import os
import glob
from tqdm import tqdm

import torch

from tensorboardX import SummaryWriter

from utils.metric import Metric


class Model(object):
    def __init__(self, criterion, device, opt, train_loader=None, test_loader=None, log_dir='./logs', resume=None):
        super(Model, self).__init__()
        # resume:
        #   1. None - it starts from scratch
        #   2. 'default' - it starts from a latest experiment
        #   3. /path/to/checkpoint.pth - it starts from the checkpoint
        #     - Checkpoint format: dictionary with {'epoch': epoch + 1, 'net': net.state_dict()}
        self._criterion = criterion
        self._device = device
        self._train_loader = train_loader
        self._train_metric = Metric(device, opt.distributed) if train_loader else None

        self._test_loader = test_loader
        self._test_metric = Metric(device, opt.distributed) if test_loader else None

        self._opt = opt

        # tensorboardX log directory
        logs = sorted(glob.glob(os.path.join(log_dir, 'exp_*')))
        exp_id = int(logs[-1].split('_')[-1]) if logs else 0
        exp_id += 0 if resume == 'default' or not logs else 1
        self._log_dir = os.path.join(log_dir, 'exp_' + str(exp_id))

        ckpt_file = sorted(glob.glob(os.path.join(self._log_dir, 'ckpt_*')))
        ckpt_file = ckpt_file[-1] if ckpt_file else ''
        self._ckpt_file = resume if resume is not None and resume != 'default' else ckpt_file

        self._save_interval = 10 if opt is None else opt.save_interval
        self._report_interval = 10 if opt is None else opt.report_interval
        self._best_record = 0.0

        if opt.local_rank == 0:
            self._summary_writer = SummaryWriter(self._log_dir)
            self._detail_writer = SummaryWriter(os.path.join(self._log_dir, 'details'))

            # Options are saved for managing hyper parameters
            with open(os.path.join(self._log_dir, 'options'), 'wt') as f:
                f.write('------------ Options -------------\n')
                for k, v in sorted(vars(opt).items()):
                    f.write('%s: %s\n' % (str(k), str(v)))
                f.write('-------------- End ----------------\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._opt.local_rank == 0:
            self._summary_writer.close()
            self._detail_writer.close()

    @staticmethod
    def prediction(x, k=5):
        with torch.no_grad():
            softmax = torch.nn.Softmax(dim=1)
            x = softmax(x)
            probs, preds = x.topk(k, dim=1, largest=True, sorted=True)

        return probs.t(), preds.t()

    def load(self, net):
        if os.path.isfile(self._ckpt_file):
            print("loading checkpoint '{}'".format(self._ckpt_file))
            ckpt = torch.load(self._ckpt_file, map_location=lambda storage, loc: storage)
            start_epoch = ckpt['epoch']
            net.module.load_state_dict(ckpt['net'])
        else:
            print("no checkpoint found at '{}'".format(self._ckpt_file))
            start_epoch = 0

        return start_epoch

    def train(self, net, optimizer, epoch, scheduler, scale_loss=None):
        net.train()
        torch.cuda.empty_cache()
        self._train_metric.reset()
        train_loss = 0.0
        train_count = 0

        scheduler.step()

        if isinstance(self._train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            self._train_loader.sampler.set_epoch(epoch)

        for idx, (imgs, labels) in tqdm(enumerate(self._train_loader)):
            imgs, labels = imgs.to(self._device), labels.to(self._device)

            outputs = net(imgs) if self._device.type == 'cuda' else net.module(imgs)

            loss = self._criterion(outputs, labels)

            optimizer.zero_grad()

            if self._opt.fp16:
                with scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()            

            optimizer.step()

            torch.cuda.synchronize()

            with torch.no_grad():
                train_loss += loss.item()
                train_count += imgs.shape[0]  # mini-batch size

                probs, preds = self.prediction(outputs)
                self._train_metric.update(preds, labels)
                top1_acc, top5_acc = self._train_metric.calculate()

                if (idx + 1) % self._report_interval == 0 and self._opt.local_rank == 0:
                    print('[Epoch:%03d, Iter:%03d] Loss: %.3f, Top 1 Accuracy: %.3f, Top 5 Accuracy: %.3f' %
                          (epoch, idx + 1, loss.item(), top1_acc, top5_acc))

        with torch.no_grad():
            if self._opt.local_rank == 0:
                topk_figs = self._train_loader.dataset.make_topk(imgs.cpu(), labels.cpu(), preds.cpu(), probs.cpu())

                for idx, topk_fig in enumerate(topk_figs):
                    self._detail_writer.add_figure('train/image_%d' % idx, topk_fig, epoch)

                if (epoch + 1) % self._save_interval == 0 and self._opt.local_rank == 0:
                    torch.save({'epoch': epoch + 1,
                                'net': net.module.state_dict()}, os.path.join(self._log_dir, 'ckpt_%05d.pth' % epoch))

            return train_loss / train_count

    def eval(self, net, epoch=None):
        net.eval()
        torch.cuda.empty_cache()
        eval_loss = 0.0
        eval_count = 0.0

        with torch.no_grad():
            self._test_metric.reset()

            for idx, (imgs, labels) in tqdm(enumerate(self._test_loader)):
                imgs, labels = imgs.to(self._device), labels.to(self._device)

                outputs = net(imgs) if self._device.type == 'cuda' else net.module(imgs)

                loss = self._criterion(outputs, labels)

                eval_loss += loss.item()
                eval_count += imgs.shape[0]  # mini-batch size
                torch.cuda.synchronize()

                probs, preds = self.prediction(outputs)
                self._test_metric.update(preds, labels)

            top1_acc, _ = self._test_metric.calculate()
            if self._opt.local_rank == 0:
                if top1_acc > self._best_record and epoch is not None:
                    torch.save({'epoch': epoch,
                                'net': net.module.state_dict()}, os.path.join(self._log_dir, 'best_top1acc_%.3f.pth'
                                                                              % top1_acc))
                    self._best_record = top1_acc

                topk_figs = self._test_loader.dataset.make_topk(imgs.cpu(), labels.cpu(), preds.cpu(), probs.cpu())
                for idx, topk_fig in enumerate(topk_figs):
                    self._detail_writer.add_figure('train/image_%d' % idx, topk_fig, 0)

            return eval_loss / eval_count

    def log(self, tags, loss, epoch=0, learning_rate=None, summary=False):
        if self._opt.local_rank != 0:
            return

        tags = [tags] if not isinstance(tags, list) else tags
        metric = {'train': self._train_metric, 'eval': self._test_metric}

        top1_acc = {}
        top5_acc = {}

        # Multi-variable scalars are generated
        for tag in tags:
            top1_acc[tag], top5_acc[tag] = metric[tag].calculate()

            print('[%s-Epoch:%03d] Loss: %.3f, Top 1 Accuracy: %.3f, Top 5 Accuracy: %.3f'
                  % (tag.capitalize(), epoch, loss[tag], top1_acc[tag], top5_acc[tag]))

            self._summary_writer.add_scalar('%s/total_loss' % tag, loss[tag], epoch)
            self._summary_writer.add_scalar('%s/top1_accuracy' % tag, top1_acc[tag], epoch)
            self._summary_writer.add_scalar('%s/top5_accuracy' % tag, top5_acc[tag], epoch)

            if tag is 'train' and learning_rate is not None:
                self._summary_writer.add_scalar('%s/learning_rate' % tag, learning_rate, epoch)

        if len(tags) > 1 and summary:
            self._summary_writer.add_scalars('total_loss', loss, epoch)
            self._summary_writer.add_scalars('top1_accuracy', top1_acc, epoch)
            self._summary_writer.add_scalars('top5_accuracy', top5_acc, epoch)
