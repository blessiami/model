import torch
import numpy as np


class PreFetcher(object):
    def __init__(self, dataset, batch_size=1, sampler=None, shuffle=False, num_workers=0, pin_memory=False,
                 drop_last=False, transforms=None):
        super(PreFetcher, self).__init__()
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
                                                  num_workers=num_workers, pin_memory=pin_memory, collate_fn=totensor,
                                                  drop_last=drop_last)
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.transforms = transforms

        self.mean = torch.tensor([m * 255 for m in self.loader.dataset.mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([s * 255 for s in self.loader.dataset.std]).cuda().view(1, 3, 1, 1)

        self.sampler = self.loader.sampler
        self.dataset = self.loader.dataset

        self.preload()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.iter)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

            self.next_input = self.next_input.float()

            if self.transforms is not None:
                self.next_input = self.transforms(self.next_input)

            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def totensor(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets