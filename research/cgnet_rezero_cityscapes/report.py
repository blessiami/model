import os
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict

import torch

from datasets.cityscapes import Cityscapes as Dataset
from datasets import transforms as transforms

from networks.SegmentationModel import EESPNet_Seg as Network

from utils.metric import Metric

from option import Option


def report(opt):
    device = torch.device(opt.device)

    transform = transforms.Compose([transforms.CenterCrop(size=tuple(opt.val_size)),
                                    transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                    transforms.ToTensor()])

    dataset = Dataset(opt.val_data, splits='val', transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    net = Network(classes=dataset.n_category, s=1.5, pretrained=opt.base_weight).to(device)

    if os.path.isfile(opt.resume if opt.resume is not None else ''):
        print("loading checkpoint '{}'".format(opt.resume))
        ckpt = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt['net'])

    net.eval()
    torch.cuda.empty_cache()

    stat = []
    metric = Metric(dataset.n_category, device, dataset.ignore_index)

    with torch.no_grad():
        for idx, mini_batch in tqdm(enumerate(loader)):
            imgs, labels = mini_batch['image'].to(device), mini_batch['label'].to(device)

            outputs = net(imgs)

            _, predictions = net.prediction(outputs)  # the maximum category index of outputs

            metric.reset()
            metric.update(predictions, labels)
            summary = metric.calculate(valid=True)

            for k in summary.keys():
                summary[k] = summary[k].cpu().numpy()

            data = OrderedDict()
            data['id'] = loader.dataset.ids[idx]
            data['mean_iou'] = summary['mean_iou']
            data['pixel_accuracy'] = summary['pixel_accuracy']
            data['mean_accuracy'] = summary['mean_accuracy']
            data['frequency_weighted_iou'] = summary['frequency_weighted_iou']

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_iou'] = summary['iou'][i]

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_precision'] = summary['precision'][i]

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_recall'] = summary['recall'][i]

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_f1_score'] = summary['f1_score'][i]

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_true_positive'] = summary['true_positive'][i]

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_false_positive'] = summary['false_positive'][i]

            for i, info in enumerate(dataset.info):
                data[info['name'] + '_false_negative'] = summary['false_negative'][i]

            stat.append(data)

        stat = pd.DataFrame(stat)
        stat.to_excel('statistics.xlsx')
        print(stat.to_string())


if __name__ == '__main__':
    report(Option().parse())
