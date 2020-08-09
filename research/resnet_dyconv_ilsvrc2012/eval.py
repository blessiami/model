import torch
import torch.utils.data.distributed

from datasets.imagenet import ImageNet as Dataset
import datasets.transforms as transforms

from networks.resnet import resnet50 as Network
from networks.smooth_cross_entropy import SmoothCrossEntropy as Loss

from utils.parallel import Parallel

from option import Option
from model import Model


def trainval(opt):
    parallel = Parallel(opt)

    device = torch.device(opt.device)

    val_transform = transforms.Compose([transforms.Resize(opt.val_resize),
                                        transforms.CenterCrop(opt.val_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=Dataset.mean, std=Dataset.std)])

    # dataset = Dataset(opt.val_data.replace('/Data', ''), splits='adversal', transform=val_transform)
    dataset = Dataset(opt.val_data, splits='val_new', transform=val_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if opt.distributed else None

    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.val_batch, sampler=sampler, shuffle=False,
                                         num_workers=opt.num_workers, pin_memory=True)

    net = Network(pretrained=True)
    net = net.to(device) if not opt.sync_bn else parallel.get('SYNC_BN')(net).to(device)

    criterion = Loss(k=dataset.n_category, eps=0.1).to(device) if opt.smooth else torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    if opt.fp16:
        net, optimizer = parallel.get('FP16_OPT')(net, optimizer, opt_level=opt.opt_level)

    net = torch.nn.DataParallel(net) if not opt.distributed else parallel.get('DDP')(net, delay_allreduce=True)

    with Model(criterion, device, opt, test_loader=loader, resume=opt.resume) as model:
        model.load(net)

        loss = {'eval': model.eval(net)}
        model.log('eval', loss)


if __name__ == '__main__':
    trainval(Option().parse())
