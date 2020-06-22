import torch
import torch.utils.data.distributed

from datasets.camvid import CamVid as Dataset
from datasets import transforms as transforms

from networks.segnet import SegNet as Network

from utils.scheduler import StepLR
from utils.parallel import Parallel
from utils.nvidia_smi import NvidiaSmi

from option import Option
from model import Model


def trainval(opt):
    parallel = Parallel(opt)

    device = torch.device(opt.device)

    train_transform = transforms.Compose([transforms.RandomScale(limit=opt.scale_limits, step=opt.scale_step),
                                          transforms.RandomCrop(size=tuple(opt.train_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                          transforms.ToTensor()])

    train_transform = transforms.Compose([#transforms.RandomScale(limit=opt.scale_limits, step=opt.scale_step),
                                          #transforms.RandomCrop(size=tuple(opt.train_size)),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                          transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.CenterCrop(size=tuple(opt.val_size)),
                                        transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                        transforms.ToTensor()])

    val_dataset = Dataset(opt.val_data, splits='val', transform=val_transform)
    train_dataset = Dataset(opt.train_data, transform=train_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if opt.distributed else None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_batch, sampler=train_sampler,
                                               shuffle=(train_sampler is None), num_workers=opt.num_workers,
                                               drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.val_batch, shuffle=False,
                                             num_workers=opt.num_workers, pin_memory=True)

    net = Network(n_classes=3)
    net = net if not opt.sync_bn else parallel.get('SYNC_BN')(net)
    net = torch.nn.DataParallel(net).to(device) if not opt.distributed else parallel.get('DDP')(net.to(device),
                                                                                                delay_allreduce=True)

    criterion = torch.nn.MSELoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    with Model(criterion, device, opt, train_loader, val_loader, resume=opt.resume) as model:
        start_epoch = model.load(net)
        start_epoch = opt.start_epoch if opt.start_epoch is not None else start_epoch
        scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma, last_epoch=start_epoch - 1)
        loss = dict()

        for epoch in range(start_epoch, opt.max_epochs):
            loss['train'] = model.train(net, optimizer, epoch, scheduler)
            loss['eval'] = model.eval(net, epoch)
            model.log(['train', 'eval'], loss, epoch, optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
#    NvidiaSmi(mem_thr=16152).chance()
    trainval(Option().parse())
