import torch
import torch.utils.data.distributed

from datasets.cityscapes import Cityscapes as Dataset
from datasets import transforms as transforms

from networks.SegmentationModel import EESPNet_Seg as Network

from utils.parallel import Parallel

from option import Option
from model import Model


def eval(opt):
    parallel = Parallel(opt)

    device = torch.device(opt.device)

    val_transform = transforms.Compose([transforms.CenterCrop(size=tuple(opt.val_size)),
                                        transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                        transforms.ToTensor()])

    val_dataset = Dataset(opt.val_data, splits='val', transform=val_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.val_batch, shuffle=False,
                                             num_workers=opt.num_workers, pin_memory=True)

    net = Network(classes=val_dataset.n_category, s=1.5, pretrained=opt.base_weight)
    net = net if not opt.sync_bn else parallel.get('SYNC_BN')(net)
    net = torch.nn.DataParallel(net).to(device) if not opt.distributed else parallel.get('DDP')(net.to(device),
                                                                                                delay_allreduce=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=val_dataset.ignore_index).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    with Model(criterion, device, opt, test_loader=val_loader, resume=opt.resume) as model:
        start_epoch = model.load(net)
        loss = dict()

        loss['eval'] = model.eval(net)
        model.log('eval', loss, start_epoch, optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    eval(Option().parse())
