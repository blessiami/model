import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .layer import SegNetDown2, SegNetDown3, SegNetUp2, SegNetUp3


class SegNet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegNetDown2(self.in_channels, 64)
        self.down2 = SegNetDown2(64, 128)
        self.down3 = SegNetDown3(128, 256)
        self.down4 = SegNetDown3(256, 512)
        self.down5 = SegNetDown3(512, 512)

        self.up5 = SegNetUp3(512, 512)
        self.up4 = SegNetUp3(512, 256)
        self.up3 = SegNetUp3(256, 128)
        self.up2 = SegNetUp2(128, 64)
        self.up1 = SegNetUp2(64, n_classes)

        self.tanh = nn.Tanh()

        # self._init_weight()

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        up1 = self.tanh(up1)

        return up1

    def _init_weight(self):
        vgg16 = models.vgg16(pretrained=True)

    # def init_vgg16_params(self, vgg16):
        blocks = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit]
            else:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit,
                         conv_block.conv3.cbr_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    @staticmethod
    def prediction(x, k=1, select='max'):
        with torch.no_grad():
            if select is 'max':
                probs, preds = torch.max(x, dim=1, keepdim=True)  # the maximum category index of outputs
            else:
                softmax = nn.Softmax2d()
                x = softmax(x)
                probs, preds = x.topk(k, dim=1, largest=True, sorted=True)

        return probs.squeeze(1), preds.squeeze(1)

