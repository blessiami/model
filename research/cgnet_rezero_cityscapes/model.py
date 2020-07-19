import os
import glob
from tqdm import tqdm

import torch

from tensorboardX import SummaryWriter

from utils.metric import Metric


class Model(object):
    def __init__(self, criterion, device, opt, train_loader=None, test_loader=None, log_dir='./logs', resume=None):
        # resume:
        #   1. None - it starts from scratch
        #   2. 'default' - it starts from a latest experiment
        #   3. /path/to/checkpoint.pth - it starts from the checkpoint
        #     - Checkpoint format: dictionary with {'epoch': epoch + 1, 'net': net.state_dict()}

        super(Model, self).__init__()
        self._criterion = criterion
        self._device = device
        self._train_loader = train_loader
        self._train_metric = Metric(train_loader.dataset.n_category, device, train_loader.dataset.ignore_index,
                                    distributed=opt.distributed) if train_loader else None

        self._test_loader = test_loader
        self._test_metric = Metric(test_loader.dataset.n_category, device, test_loader.dataset.ignore_index,
                                   distributed=opt.distributed) if test_loader else None

        self._opt = opt

        self._train_record = False

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

        # Options are saved for managing hyper parameters
        if opt.local_rank == 0:
            self._summary_writer = SummaryWriter(self._log_dir)
            self._detail_writer = SummaryWriter(os.path.join(self._log_dir, 'details'))

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

    @staticmethod
    def prediction(x, k=1, select='max'):
        with torch.no_grad():
            if select is 'max':
                probs, preds = torch.max(x, dim=1, keepdim=True)  # the maximum category index of outputs
            else:
                softmax = torch.nn.Softmax2d()
                x = softmax(x)
                probs, preds = x.topk(k, dim=1, largest=True, sorted=True)

        return probs.squeeze(1), preds.squeeze(1)

    def train(self, net, optimizer, epoch, scheduler):
        net.train()
        torch.cuda.empty_cache()
        self._train_metric.reset()
        train_loss = 0.0
        train_count = 0

        if isinstance(self._train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            self._train_loader.sampler.set_epoch(epoch)

        for idx, mini_batch in tqdm(enumerate(self._train_loader)):
            scheduler.step(epoch=None if self._opt.poly_lr else epoch)

            imgs, labels = mini_batch['image'].to(self._device), mini_batch['label'].to(self._device)

            outputs = net(imgs) if self._device.type == 'cuda' else net.module(imgs)
            loss = self._criterion(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()
                train_count += imgs.shape[0]  # mini-batch size

                _, predictions = self.prediction(outputs) # the maximum category index of outputs
                self._train_metric.update(predictions, labels)
                metric = self._train_metric.calculate()

                if (idx + 1) % self._report_interval == 0:
                    print('[Epoch:%03d, Iter:%03d] Learning Rate: %.7f, Loss: %.3f, Pixel Accuracy: %.3f, mIoU: %.3f' %
                          (epoch, idx + 1, optimizer.param_groups[0]['lr'], loss.item(), metric['pixel_accuracy'],
                           metric['mean_iou']))

        with torch.no_grad():
            if self._opt.local_rank == 0:
                if (epoch + 1) % self._save_interval == 0:
                    torch.save({'epoch': epoch + 1,
                                'net': net.module.state_dict()}, os.path.join(self._log_dir, 'ckpt_%05d.pth' % epoch))

                if self._train_record:
                    grid_imgs = self._train_loader.dataset.make_grid(imgs, labels, predictions)

                    for idx, grid_img in enumerate(grid_imgs):
                        self._detail_writer.add_image('train/image_%d' % idx, grid_img, epoch)

                    self._train_record = False

            return train_loss / train_count

    def eval(self, net, epoch=None):
        net.eval()
        torch.cuda.empty_cache()
        eval_loss = 0.0
        eval_count = 0

        with torch.no_grad():
            self._test_metric.reset()

            for idx, mini_batch in tqdm(enumerate(self._test_loader)):
                imgs, labels = mini_batch['image'].to(self._device), mini_batch['label'].to(self._device)

                outputs = net(imgs) if self._device.type == 'cuda' else net.module(imgs)
                loss = self._criterion(outputs, labels.long())

                eval_loss += loss.item()
                eval_count += imgs.shape[0]  # mini-batch size

                _, predictions = self.prediction(outputs)  # the maximum category index of outputs
                self._test_metric.update(predictions, labels)

            metric = self._test_metric.calculate(valid=True)

            if self._opt.local_rank == 0:
                if metric['mean_iou'] > self._best_record and epoch is not None:
                    torch.save({'epoch': epoch,
                                'net': net.module.state_dict()}, os.path.join(self._log_dir, 'best_miou_%.3f.pth' % metric['mean_iou']))
                    self._best_record = metric['mean_iou']

                    grid_imgs = self._test_loader.dataset.make_grid(imgs, labels, predictions)
                    for idx, grid_img in enumerate(grid_imgs):
                        self._detail_writer.add_image('validation/image_%d' % idx, grid_img, 0)

                    self._train_record = True

            return eval_loss / eval_count

    def log(self, tags, loss, epoch=0, learning_rate=None):
        if self._opt.local_rank != 0:
            return

        tags = [tags] if not isinstance(tags, list) else tags

        metrics = {'train': self._train_metric, 'eval': self._test_metric}
        loaders = {'train': self._train_loader, 'eval': self._test_loader}

        for tag in tags:
            metric = metrics[tag].calculate(valid=True)
            info = loaders[tag].dataset.info

            print('[%s-Epoch:%03d] Loss: %.3f, Pixel Acc: %.3f, Mean Acc: %.3f, mIoU: %.3f, FW IoU: %.3f'
                  % (tag.capitalize(), epoch, loss[tag], metric['pixel_accuracy'], metric['mean_accuracy'],
                     metric['mean_iou'], metric['frequency_weighted_iou']))

            for i, item in enumerate(info):
                print('  -%-17s - iou: %.3f, precision: %.3f, recall: %.3f, f1_score: %.3f, tp: %d, fp: %d, fn: %d'
                      % (item['name'], metric['iou'][i], metric['precision'][i], metric['recall'][i],
                         metric['f1_score'][i], metric['true_positive'][i], metric['false_positive'][i], metric['false_negative'][i]))

                self._summary_writer.add_scalar('%s_detail/%s-iou' % (tag, item['name']), metric['iou'][i], epoch)

            self._summary_writer.add_scalar('%s/total_loss' % tag, loss[tag], epoch)
            self._summary_writer.add_scalar('%s/pixel_accuracy' % tag, metric['pixel_accuracy'], epoch)
            self._summary_writer.add_scalar('%s/mean_accuracy' % tag, metric['mean_accuracy'], epoch)
            self._summary_writer.add_scalar('%s/mean_iou' % tag, metric['mean_iou'], epoch)
            self._summary_writer.add_scalar('%s/frequency_weighted_iou' % tag, metric['frequency_weighted_iou'], epoch)

            if tag is 'train' and learning_rate is not None:
                self._summary_writer.add_scalar('%s/learning_rate' % tag, learning_rate, epoch)

            if tag is 'eval':
                self._summary_writer.add_scalar('%s/best_miou' % tag, self._best_record, epoch)
