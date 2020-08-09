import os
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('Matplotlib: No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as datasets

from .imagenet_category import category


class ImageNet(datasets.ImageFolder):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, path, splits='train', transform=None):
        root = os.path.join(path, splits)
        super(ImageNet, self).__init__(root, transform=transform)
        self.n_category = len(self.classes)

    def _abnormalize(self, img):
        with torch.no_grad():
            img = img.detach().cpu()
            img = np.array(img).astype(np.float32).transpose(1, 2, 0)
            img = ((img * self.std) + self.mean) * 255.0

        return img.astype(dtype=np.uint8)

    def make_topk(self, imgs, labels, preds, probs):
        figs = []
        colors = ['#DEEBF7', '#BDD7EE', '#9DC3E6', '#2E75B6', '#1F4E79']

        for i, label in enumerate(labels):
            title = category[label.item()].split(',')[0]

            img = self._abnormalize(imgs[i])

            fig, (img_ax, bar_ax) = plt.subplots(2, 1, frameon=False)
            img_ax.set_title(title)
            img_ax.axis('off')
            img_ax.set_position([0.1, 0.3, 0.8, 0.65])
            img_ax.imshow(img, interpolation='bilinear', aspect='auto')

            rate = np.flip(probs[:, i].numpy(), 0)
            name = tuple([category[x].split(',')[0] for x in np.flip(preds[:, i].numpy(), 0)])
            n_bars = np.arange(preds.size(0))

            bars = bar_ax.barh(n_bars, rate, align='center')

            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
                bar_ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, '%.3f' % rate[i], va='center')

            bar_ax.set_yticks(n_bars)
            bar_ax.set_yticklabels(name)
            bar_ax.set_position([0.2, 0.05, 0.7, 0.25])
            bar_ax.set_xlim(0, 1)

            figs.append(fig)

        return figs

    def show(self, imgs, idx=0):
        from PIL import Image
        img = self._abnormalize(imgs[idx])
        img = Image.fromarray(img, 'RGB')
        img.show()
