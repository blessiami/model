import torch
from torchvision.transforms.transforms import *


class Lighting(object):
    def __init__(self, alpha=0.1, eig_val=torch.Tensor([0.2175, 0.0188, 0.0045]),
                 eig_vec=torch.Tensor([[-0.5675,  0.7192,  0.4009],
                                       [-0.5808, -0.0045, -0.8140],
                                       [-0.5836, -0.6948,  0.4203]])):
        self._alpha = alpha
        self._eig_val = eig_val
        self._eig_vec = eig_vec

    def __call__(self, img):
        if self._eig_val.device != img.device:
            self._eig_val = self._eig_val.to(img.device)
            self._eig_vec = self._eig_vec.to(img.device)

        if self._alpha != 0:
            alpha = img.new().resize_(3).normal_(0, self._alpha)
            rgb = self._eig_vec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3))\
                .mul(self._eig_val.view(1, 3).expand(3, 3)).sum(1).squeeze()
            img = img.add(rgb.view(3, 1, 1).expand_as(img))

        return img
