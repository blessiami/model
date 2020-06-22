import os
import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image


class CamVid(Dataset):
    n_category = 12

    mean = (0.485, 0.456, 0.406)  # (123, 116, 104)
    std = (0.229, 0.224, 0.225)

    # https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Scripts/test_segmentation_camvid.py
    info = [{'idx':  0, 'name': 'sky',         'palette': [128, 128, 128]},
            {'idx':  1, 'name': 'building',    'palette': [128,   0,   0]},
            {'idx':  2, 'name': 'pole',        'palette': [192, 192, 128]},
            {'idx':  3, 'name': 'road',        'palette': [128,  64, 128]},
            {'idx':  4, 'name': 'pavement',    'palette': [ 60,  40, 222]},
            {'idx':  5, 'name': 'tree',        'palette': [128, 128,   0]},
            {'idx':  6, 'name': 'sign_symbol', 'palette': [192, 128, 128]},
            {'idx':  7, 'name': 'fence',       'palette': [ 64,  64, 128]},
            {'idx':  8, 'name': 'car',         'palette': [ 64,   0, 128]},
            {'idx':  9, 'name': 'pedestrian',  'palette': [ 64,  64,   0]},
            {'idx': 10, 'name': 'bicyclist',   'palette': [  0, 128, 192]},
            {'idx': 11, 'name': 'unlabelled',  'palette': [  0,   0,   0]}]

    ignore_index = 255
    ignore_palette = np.asarray([255, 255, 255])

    id_maps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    width = 480
    height = 360

    def __init__(self, path, splits='train', transform=None):
        super(CamVid, self).__init__()
        self._transform = transform

        if isinstance(splits, str):
            splits = [splits]

        self.ids = []
        self._imgs = []
        self._segs = []

        for split in splits:
            imgs = [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(os.path.join(path, split))
                    for filename in filenames if filename.endswith('.png')]
            self._imgs.extend(imgs)
            self.ids.extend([os.path.splitext(img.split(os.sep)[-1])[0] for img in imgs])
            self._segs.extend([os.path.join(os.path.join(path, split + 'annot'), os.path.basename(img)) for img in imgs])

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        ########
        idx = 0
        ########

        img = Image.open(self._imgs[idx]).convert('RGB')
        seg = np.array(Image.open(self._segs[idx]), dtype=np.uint8)

        for i, id in enumerate(self.id_maps):
            seg[seg == i] = id

        seg = Image.fromarray(seg)

        output = {'image': img, 'label': seg}

        if self._transform is not None:
            output = self._transform(output)

        return output

    def _abnormalize(self, img):
        with torch.no_grad():
            img = img.detach().cpu()
            img = np.array(img).astype(np.float32).transpose(1, 2, 0)
            img = ((img * self.std) + self.mean) * 255.0

        return img.astype(dtype=np.uint8)

    def _seg2rgb(self, seg):
        with torch.no_grad():
            seg = seg.detach().cpu()
            seg = np.array(seg).astype(np.float32)

            r = seg.copy()
            g = seg.copy()
            b = seg.copy()

            rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

            # Class pixels are painted
            for i in range(self.n_category):
                r[seg == i] = self.info[i]['palette'][0]
                g[seg == i] = self.info[i]['palette'][1]
                b[seg == i] = self.info[i]['palette'][2]

            # Ignore pixels are painted
            r[seg == self.ignore_index] = self.ignore_palette[0]
            g[seg == self.ignore_index] = self.ignore_palette[1]
            b[seg == self.ignore_index] = self.ignore_palette[2]

            rgb[:, :, 0] = r
            rgb[:, :, 1] = g
            rgb[:, :, 2] = b

        return rgb

    def make_grid(self, imgs, segs, predictions):
        batch_size = imgs.shape[0]
        grid_imgs = []

        for i in range(batch_size):
            img = self._abnormalize(imgs[i])
            seg = self._abnormalize(segs[i]).clip(0, 255)
            prediction = self._abnormalize(predictions[i]).clip(0, 255)
            # seg = self._seg2rgb(segs[i])
            # prediction = self._seg2rgb(predictions[i])
            grid_imgs.append(np.concatenate((img, seg, prediction), axis=1).transpose(2, 0, 1))

        return grid_imgs
