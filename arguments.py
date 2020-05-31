import argparse
import os

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Model related arguments
        parser.add_argument('--id', default='toy',
                            help="a name for identifying the model")

        # Data related arguments
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=32, type=int,
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_vis', default=40, type=int,
                            help='number of images to evalutate')

        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./ckpt',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='data/train.csv')
        parser.add_argument('--list_val',
                            default='data/val.csv')
        parser.add_argument('--dup_trainset', default=100, type=int,
                            help='duplicate so that one epoch has more iters')

        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_toy', default=1e-3, type=float, help='LR')
        
        parser.add_argument('--lr_steps',
                            nargs='+', type=int, default=[40, 60],
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')

        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

     def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        args.batch_size = args.num_gpus * args.batch_size_per_gpu
        args.ckpt = os.path.join(args.ckpt, args.id)
        args.best_err = float("inf")
        self.print_arguments(args)
        return args
