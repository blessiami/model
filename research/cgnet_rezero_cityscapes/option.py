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
        self._parser.add_argument('--train_batch', type=int, default=4, help='The number of batch sizes for training')
        self._parser.add_argument('--train_size', nargs='+', type=int, default=[768, 768],
                                  help='Training image size [width, height]')
        self._parser.add_argument('--train_data', type=str, default='/home/ljy/work/data/cityscapes',
                                  help='Training dataset path')
        self._parser.add_argument('--base_weight', type=str, default='./weight/espnetv2_s_1.5.pth',
                                  help='Pretrained base network files or None')
        self._parser.add_argument('--start_epoch', type=int, default=None, help='Use an explitcit point to restart')
        self._parser.add_argument('--max_epochs', type=int, default=300, help='Maximum epochs')
        self._parser.add_argument('--learning_rate', type=float, default=5e-4, help='Parameter update rates')
        self._parser.add_argument('--save_interval', type=int, default=10, help='Epochs for saving checkpoints')
        self._parser.add_argument('--report_interval', type=int, default=10,
                                  help='Iterations for printing out training statuses')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for parameter updates')
        self._parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 regularization parameters')

        # scheduler options
        self._parser.add_argument("--poly_lr", action='store_true', default=False,
                                  help='Turning on polynomial scheduler (default: step scheduler) for learning rate')
        # StepLR
        self._parser.add_argument('--step_size', type=int, default=20, help='Steps for updating learning rates')
        self._parser.add_argument('--gamma', type=float, default=0.94, help='Scheduler learning rate decay factor')
        # PolyLR
        self._parser.add_argument('--power', type=float, default=0.9, help='Polynomial power for learning rate decay')
        self._parser.add_argument('--max_iters', type=int, default=120000, help='Maximum iteration for using PolyLR')
        # CosineLR
        self._parser.add_argument('--max_step', type=int, default=120,
                                  help='Maximum epochs for calculating cosine learning rates')
        self._parser.add_argument('--warm_start', type=int, default=5,
                                  help='Initial epochs to increase learning rates linearly')

        # train dataset augmentations
        self._parser.add_argument('--scale_limits', nargs='+', type=float, default=[0.5, 2.0],
                                  help='Image rescale lower bound')
        self._parser.add_argument('--scale_step', type=float, default=0.25, help='Image rescale interval')

        # validation options
        self._parser.add_argument('--val_batch', type=int, default=4, help='The number of batch sizes for validation')
        self._parser.add_argument('--val_size', nargs='+', type=int, default=[2048, 1024],
                                  help='Validation image size [width, height]')
        self._parser.add_argument('--val_data', type=str, default='/home/ljy/work/data/cityscapes',
                                  help='Validation dataset path')
        self._parser.add_argument('--val_ensemble', action='store_true', default=False,
                                  help='Turning on ensemble during validation')
        self._parser.add_argument('--val_scales', nargs='+', type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                                  help='Image size scales for validation when val_ensemble=True')
        self._parser.add_argument('--val_flip', action='store_true', default=False,
                                  help='This decides to use horizontal flip when val_ensemble=True')
        self._parser.add_argument('--val_merge', type=str, default='mean',
                                  help='A merging method when val_ensemble=True')

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
        print('------------ Options -------------')
        for k, v in sorted(vars(self._opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self._opt
