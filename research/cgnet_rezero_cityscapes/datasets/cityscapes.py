import os
import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image


class Cityscapes(Dataset):
    n_category = 19
    mean = (0.485, 0.456, 0.406)  # (123, 116, 104)
    std = (0.229, 0.224, 0.225)

    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    info = [{'idx':  0, 'name': 'road',          'palette': [128,  64, 128]},
            {'idx':  1, 'name': 'sidewalk',      'palette': [244,  35, 232]},
            {'idx':  2, 'name': 'building',      'palette': [ 70,  70,  70]},
            {'idx':  3, 'name': 'wall',          'palette': [102, 102, 156]},
            {'idx':  4, 'name': 'fence',         'palette': [190, 153, 153]},
            {'idx':  5, 'name': 'pole',          'palette': [153, 153, 153]},
            {'idx':  6, 'name': 'traffic_light', 'palette': [250, 170,  30]},
            {'idx':  7, 'name': 'traffic_sign',  'palette': [220, 220,   0]},
            {'idx':  8, 'name': 'vegetation',    'palette': [107, 142,  35]},
            {'idx':  9, 'name': 'terrain',       'palette': [152, 251, 152]},
            {'idx': 10, 'name': 'sky',           'palette': [ 70, 130, 180]},
            {'idx': 11, 'name': 'person',        'palette': [220,  20,  60]},
            {'idx': 12, 'name': 'rider',         'palette': [255,   0,   0]},
            {'idx': 13, 'name': 'car',           'palette': [  0,   0, 142]},
            {'idx': 14, 'name': 'truck',         'palette': [  0,   0,  70]},
            {'idx': 15, 'name': 'bus',           'palette': [  0,  60, 100]},
            {'idx': 16, 'name': 'train',         'palette': [  0,  80, 100]},
            {'idx': 17, 'name': 'motorcycle',    'palette': [  0,   0, 230]},
            {'idx': 18, 'name': 'bicycle',       'palette': [119,  11,  32]}]

    ignore_index = 255
    ignore_palette = np.asarray([0, 0, 0])
    id_maps = [255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255,
               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18]
    license_plate_id = -1

    width = 2048
    height = 1024

    def __init__(self, path, splits='train', transform=None):
        super(Cityscapes, self).__init__()
        self._img_path = os.path.join(path, 'leftImg8bit_trainvaltest', 'leftImg8bit')
        self._seg_path = os.path.join(path, 'gtFine_trainvaltest', 'gtFine')
        self._transform = transform

        if isinstance(splits, str):
            splits = [splits]

        self.ids = []
        self._imgs = []
        self._segs = []

        for split in splits:
            imgs = [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(os.path.join(self._img_path, split))
                    for filename in filenames if filename.endswith('.png')]
            imgs.sort()

            self._imgs.extend(imgs)
            self.ids.extend([os.path.splitext(img.split(os.sep)[-1])[0] for img in imgs])
            self._segs.extend([os.path.join(os.path.join(self._seg_path, split), img.split(os.sep)[-2],
                                            os.path.basename(img)[:-15] + 'gtFine_labelIds.png') for img in imgs])

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        img = Image.open(self._imgs[idx]).convert('RGB')
        raw = np.array(Image.open(self._segs[idx]), dtype=np.uint8)
        seg = np.ones_like(raw) * self.ignore_index

        seg[seg == self.license_plate_id] = self.ignore_index

        for i, id in enumerate(self.id_maps):
            seg[raw == i] = id

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
            seg = self._seg2rgb(segs[i])
            prediction = self._seg2rgb(predictions[i])
            grid_imgs.append(np.concatenate((img, seg, prediction), axis=1).transpose(2, 0, 1))

        return grid_imgs

    def show(self, img):
        from PIL import Image

        img = self._abnormalize(img)
        img = Image.fromarray(img)
        img.show()
