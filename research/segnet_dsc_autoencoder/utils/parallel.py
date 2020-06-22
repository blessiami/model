import os

import torch
import torch.distributed

try:
    import apex
    from apex.parallel import DistributedDataParallel
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


torch.backends.cudnn.benchmark = True


class Parallel(object):
    module = {'DDP': DistributedDataParallel,
              'SYNC_BN': apex.parallel.convert_syncbn_model,
              'FP16_OPT': amp.initialize if hasattr(amp, 'initialize') else None,
              'FP16_LOSS': amp.scale_loss if hasattr(amp, 'scale_loss') else None}

    def __init__(self, opt):
        super(Parallel, self).__init__()
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1 if 'WORLD_SIZE' in os.environ else False

        if opt.distributed:
            torch.cuda.set_device(opt.local_rank) if opt.fixed_rank is False else torch.cuda.set_device(0)

            world_size = -1 if opt.init_method == 'env://' else int(os.environ['WORLD_SIZE'])
            rank = -1 if opt.init_method == 'env://' else int(opt.local_rank)

            torch.distributed.init_process_group(backend=opt.backend, init_method=opt.init_method,
                                                 world_size=world_size, rank=rank)
            opt.world_size = torch.distributed.get_world_size()
            assert opt.device == 'cuda', "Distributed mode requires running with CUDA."

    def get(self, key):
        return self.module[key]
