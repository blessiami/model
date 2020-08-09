import argparse


class Option(object):
    def __init__(self):
        super(Option, self).__init__()
        self._parser = argparse.ArgumentParser()

        # basic options
        self._parser.add_argument('--num_workers', type=int, default=0, help='The number of workers')
        self._parser.add_argument('--device', type=str, default='cuda',
                                  help='it must be cpu or cuda to select processing devices')
        self._parser.add_argument('--resume', type=str, default=None,
                                  help='''1. None - it starts from scratch
                                  2. 'default' - it starts from a latest experiment
                                  3. /path/to/checkpoint.pth - it starts from the checkpoint
                                  - Checkpoint format: dictionary with {'epoch': epoch + 1, 'net': net.state_dict()}''')

        # train options
        self._parser.add_argument('--train_batch', type=int, default=2, help='The number of batch sizes for training')
        self._parser.add_argument('--train_size', type=int, default=224, help='Training image size [width, height]')
        self._parser.add_argument('--train_data', type=str, default='/home/ljy/work/data/ilsvrc2012/Data',
                                  help='Training dataset path')
        self._parser.add_argument('--start_epoch', type=int, default=None, help='Use an explitcit point to restart')
        self._parser.add_argument('--max_epochs', type=int, default=120, help='Maximum epochs')
        self._parser.add_argument('--learning_rate', type=float, default=0.1, help='Parameter update rates')
        self._parser.add_argument('--save_interval', type=int, default=1, help='Epochs for saving checkpoints')
        self._parser.add_argument('--report_interval', type=int, default=10,
                                  help='Iterations for printing out training statuses')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for parameter updates')
        self._parser.add_argument('--weight_decay', type=float, default=4e-5, help='L2 regularization parameters')

        # lr_scheduler.StepLR options
        self._parser.add_argument('--step_size', type=int, default=30, help='Steps for updating learning rates')
        self._parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler learning rate decay factor')

        # lr_scheduler.CosineLR options
        self._parser.add_argument('--max_step', type=int, default=120,
                                  help='Maximum epochs for calculating cosine learning rates')
        self._parser.add_argument('--warm_start', type=int, default=5,
                                  help='Initial epochs to increase learning rates linearly')

        # label smoothing options
        self._parser.add_argument("--smooth", action='store_true', default=False,
                                  help='It enables label smoothing to reduce empirical gap')

        # validation options
        self._parser.add_argument('--val_batch', type=int, default=1024, help='The number of batch sizes for validation')
        self._parser.add_argument('--val_resize', type=int, default=256, help='Validation resize for center cropping')
        self._parser.add_argument('--val_size', type=int, default=224, help='Validation image size for transforms')
        self._parser.add_argument('--val_data', type=str, default='/home/ljy/work/data/ilsvrc2012/Data',
                                  help='Validation dataset path')

        # fp16 options
        self._parser.add_argument("--fp16", action='store_true', default=False,
                                  help='Turning on mixed point training')
        self._parser.add_argument('--opt_level', type=str, default='O1', help='The optimization levels')

        # added for distributed
        self._parser.add_argument("--local_rank", type=int, default=0,
                                  help='The rank obtained from torch.distributed.launch modules')
        self._parser.add_argument("--sync_bn", action='store_true', default=False,
                                  help='Turning on synchronous batchnorm')
        self._parser.add_argument("--fixed_rank", action='store_true', default=False,
                                  help='It launches multi-process training and validation for single GPU')
        self._parser.add_argument("--backend", type=str, default='nccl',
                                  help='It decides backend channels for distributed computing')
        self._parser.add_argument("--init_method", type=str, default='env://',
        # self._parser.add_argument("--init_method", type=str, default='tcp://10.230.144.202:6000',
                                  help='It selects initialization methods for distributed computing')

        self._opt = self._parser.parse_args()

        assert self._opt.device == 'cpu' or self._opt.device == 'cuda'

    def parse(self):
        if self._opt.local_rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(vars(self._opt).items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        return self._opt
