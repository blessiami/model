import torch.nn as nn


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(Conv2DBatchNormRelu, self).__init__()

        if int(in_channels) == int(n_filters):
            groups = int(in_channels)
        else:
            groups = 1

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, groups=groups,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class SegNetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetDown2, self).__init__()
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetDown3, self).__init__()
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class SegNetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegNetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs
