import torch
import torch.utils.data.distributed

from datasets.imagenet import ImageNet as Dataset
import datasets.transforms as transforms
from datasets.prefetcher import PreFetcher

from networks.resnet import resnet50 as Network
from networks.smooth_cross_entropy import SmoothCrossEntropy as Loss

from utils.scheduler import CosineLR
# from utils.scheduler import StepLR
from utils.parallel import Parallel

from option import Option
from model import Model


def trainval(opt):
    parallel = Parallel(opt)

    device = torch.device(opt.device)

    train_transform = transforms.Compose([transforms.RandomResizedCrop(opt.train_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Lighting(),
                                          transforms.Normalize(mean=Dataset.mean, std=Dataset.std)])

    val_transform = transforms.Compose([transforms.Resize(opt.val_resize),
                                        transforms.CenterCrop(opt.val_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=Dataset.mean, std=Dataset.std)])

    train_dataset = Dataset(opt.train_data, transform=train_transform)
    val_dataset = Dataset(opt.val_data, splits='val_new', transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if opt.distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if opt.distributed else None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_batch, sampler=train_sampler,
                                               shuffle=(train_sampler is None), num_workers=opt.num_workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.val_batch, sampler=val_sampler, shuffle=False,
                                             num_workers=opt.num_workers, pin_memory=True)

    net = Network()
    net = net.to(device) if not opt.sync_bn else parallel.get('SYNC_BN')(net).to(device)

    criterion = Loss(k=train_dataset.n_category, eps=0.1).to(device) if opt.smooth else torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.weight_decay(opt.weight_decay), lr=opt.learning_rate, momentum=opt.momentum)
    # optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
    #                             weight_decay=opt.weight_decay)

    if opt.fp16:
        net, optimizer = parallel.get('FP16_OPT')(net, optimizer, opt_level=opt.opt_level)

    net = torch.nn.DataParallel(net) if not opt.distributed else parallel.get('DDP')(net, delay_allreduce=True)

    with Model(criterion, device, opt, train_loader, val_loader, resume=opt.resume) as model:
        start_epoch = model.load(net)
        start_epoch = opt.start_epoch if opt.start_epoch is not None else start_epoch
        scheduler = CosineLR(optimizer, max_step=opt.max_step, warm_start=opt.warm_start, last_epoch=start_epoch - 1)
        # scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma, last_epoch=start_epoch - 1)

        loss = dict()

        for epoch in range(start_epoch, opt.max_epochs):
            loss['train'] = model.train(net, optimizer, epoch, scheduler, parallel.get('FP16_LOSS'))
            loss['eval'] = model.eval(net, epoch) 
            model.log(['train', 'eval'], loss, epoch, optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    trainval(Option().parse())
