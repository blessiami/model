import random
import math

import numpy as np
from PIL import Image, ImageOps

import torch


class Compose(object):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img, 'label': mask}


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, sample):
        # numpy image to torch image: H x W x C to C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        seg = np.array(sample['label']).astype(np.float32)

        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).float()

        return {'image': img, 'label': seg}


class Scale(object):
    def __init__(self, width, height, rate=None):
        super(Scale, self).__init__()
        self._scale = (width, height)
        self._rate = rate

    def __call__(self, sample):
        img = sample['image']
        seg = sample['label']

        assert img.size == seg.size

        if self._rate is not None:
            self._scale = (int(self._rate * img.size[0]), int(self._rate * img.size[1]))

        img, seg = img.resize(self._scale, Image.BILINEAR), seg.resize(self._scale, Image.NEAREST)

        return {'image': img, 'label': seg}


class RandomScale(object):
    def __init__(self, limit, step):
        super(RandomScale, self).__init__()
        self._scale = np.arange(limit[0], limit[1], step)

    def __call__(self, sample):
        img = sample['image']
        seg = sample['label']

        assert img.size == seg.size

        scale = self._scale[random.randint(0, self._scale.size - 1)]

        w, h = int(scale * img.size[0]), int(scale * img.size[1])

        img, seg = img.resize((w, h), Image.BILINEAR), seg.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': seg}


class RandomCrop(object):
    def __init__(self, size, ignore_idx=255):
        super(RandomCrop, self).__init__()
        self._size = size
        self._ignore_idx = ignore_idx

    def __call__(self, sample):
        img, seg = sample['image'], sample['label']

        img_width, img_height = img.size

        x = random.randint(0, max(img_width - self._size[0], 0))
        y = random.randint(0, max(img_height - self._size[1], 0))

        img = img.crop((x, y, min(x + self._size[0], img_width), min(y + self._size[1], img_height)))
        seg = seg.crop((x, y, min(x + self._size[0], img_width), min(y + self._size[1], img_height)))

        border = (math.floor((self._size[0] - img.size[0]) / 2), math.floor((self._size[1] - img.size[1]) / 2),
                  math.ceil((self._size[0] - img.size[0]) / 2), math.ceil((self._size[1] - img.size[1]) / 2))

        mean_img = np.array(img).mean(axis=(0, 1)).astype('int')

        img = ImageOps.expand(img, border=border, fill=(mean_img[0], mean_img[1], mean_img[2]))
        seg = ImageOps.expand(seg, border=border, fill=self._ignore_idx)

        return {'image': img, 'label': seg}


class RandomCropResize(object):
    def __init__(self, area):
        super(RandomCropResize, self).__init__()
        self._area = area

    def __call__(self, sample):
        img, seg = sample['image'], sample['label']
        w, h = img.size

        if random.random() < 0.5:
            x = random.randint(0, self._area)
            y = random.randint(0, self._area)

            img = img.crop((x, y, w - x, h - y))
            seg = seg.crop((x, y, w - x, h - y))

            img, seg = img.resize((w, h), Image.BILINEAR), seg.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': seg}

class RandomHorizontalFlip(object):
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__()

    def __call__(self, sample):
        img = sample['image']
        seg = sample['label']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': seg}


class CenterCrop(object):
    def __init__(self, size, ignore_idx=255):
        super(CenterCrop, self).__init__()
        self._size = size
        self._ignore_idx = ignore_idx

    def __call__(self, sample):
        img = sample['image']
        seg = sample['label']

        assert img.size == seg.size

        x = max(math.floor((img.size[0] - self._size[0]) / 2), 0)
        y = max(math.floor((img.size[1] - self._size[1]) / 2), 0)

        img = img.crop((x, y, x + min(self._size[0], img.size[0]), y + min(self._size[1], img.size[1])))
        seg = seg.crop((x, y, x + min(self._size[0], img.size[0]), y + min(self._size[1], img.size[1])))

        border = (math.floor((self._size[0] - img.size[0]) / 2), math.floor((self._size[1] - img.size[1]) / 2),
                  math.ceil((self._size[0] - img.size[0]) / 2), math.ceil((self._size[1] - img.size[1]) / 2))

        mean_img = np.array(img).mean(axis=(0, 1)).astype('int')

        img = ImageOps.expand(img, border=border, fill=(mean_img[0], mean_img[1], mean_img[2]))
        seg = ImageOps.expand(seg, border=border, fill=self._ignore_idx)

        return {'image': img, 'label': seg}
