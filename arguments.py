import argparse
import os

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Model related arguments
        parser.add_argument('--id', default='',
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
