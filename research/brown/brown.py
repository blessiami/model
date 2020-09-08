import torch
import torch.nn as nn


class Brown(nn.Module):
    def __init__(self, p=0.2):
        super(Brown, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        self.p = p * 100

    def forward(self, input):
        if not self.training:
            return input

        x = input.clone()
        avg = self.pad(input)
        avg = self.avg_pool(avg)

        dir = torch.randint_like(input, low=0, high=8)
        prob = torch.randint_like(input, low=1, high=100)
        pos = prob <= self.p

        lt = (dir == 0) * pos
        x[..., :-1, :-1][lt[..., 1:, 1:]] = input[..., 1:, 1:][lt[..., 1:, 1:]]
        x[..., 1:, 1:][lt[..., 1:, 1:]] = avg[..., 1:, 1:][lt[..., 1:, 1:]]

        mt = (dir == 1) * pos
        x[..., :-1, :][mt[..., 1:, :]] = input[..., 1:, :][mt[..., 1:, :]]
        x[..., 1:, :][mt[..., 1:, :]] = avg[..., 1:, :][mt[..., 1:, :]]

        rt = (dir == 2) * pos
        x[..., :-1, 1:][rt[..., 1:, :-1]] = input[..., 1:, :-1][rt[..., 1:, :-1]]
        x[..., 1:, :-1][rt[..., 1:, :-1]] = avg[..., 1:, :-1][rt[..., 1:, :-1]]

        lm = (dir == 3) * pos
        x[..., :, :-1][lm[..., :, 1:]] = input[..., :, 1:][lm[..., :, 1:]]
        x[..., :, 1:][lm[..., :, 1:]] = avg[..., :, 1:][lm[..., :, 1:]]

        mm = (dir == 4) * pos
        x[mm] = avg[mm]

        rm = (dir == 5) * pos
        x[..., :, 1:][rm[..., :, :-1]] = input[..., :, :-1][rm[..., :, :-1]]
        x[..., :, :-1][rm[..., :, :-1]] = avg[..., :, :-1][rm[..., :, :-1]]

        lb = (dir == 6) * pos
        x[..., 1:, :-1][lb[..., :-1, 1:]] = input[..., :-1, 1:][lb[..., :-1, 1:]]
        x[..., :-1, 1:][lb[..., :-1, 1:]] = avg[..., :-1, 1:][lb[..., :-1, 1:]]

        mb = (dir == 7) * pos
        x[..., 1:, :][mb[..., :-1, :]] = input[..., :-1, :][mb[..., :-1, :]]
        x[..., :-1, :][mb[..., :-1, :]] = avg[..., :-1, :][mb[..., :-1, :]]

        rb = (dir == 8) * pos
        x[..., 1:, 1:][rb[..., :-1, :-1]] = input[..., :-1, :-1][rb[..., :-1, :-1]]
        x[..., :-1, :-1][rb[..., :-1, :-1]] = avg[..., :-1, :-1][rb[..., :-1, :-1]]

        return x


if __name__ == '__main__':
    input = torch.randn(1, 1, 5, 5)
    m = Brown()
    # m = nn.Dropout(0.2)
    output = m(input)

    print(input)
    print(output)