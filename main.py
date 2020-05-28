from .arguments import ArgParser
import torch

import random

from .models import NetBuilder
from .models import NetWrapper
from .dataset import ToyDataset


def build_model(args):
    # Net Builder
    builder = NetBuilder()
    net_toy = NetBuilder.build_toy_net(args.output_dim)
    nets = (net_toy)
    crit = builder.build_criterion(arch=args.loss)

    # Net Wrapper
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(
        netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    return netWrapper


def create_optimizer(nets, args):
    (net_toy) = nets
    param_groups = [{'params': net_toy.fc.parameters(), 'lr': args.lr}]
    return torch.optim.SGD(
        param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def create_data_loaders(args):
    # Dataset and Loader
    dataset_train = ToyDataset(
        args.train_list, args, split='train')
    dataset_val = ToyDataset(
        args.val_list, args, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)

    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    return loader_train, loader_val


def main(args):
    # Model
    model = build_model(args)

    # Set up optimizer
    optimizer = create_optimizer(model, args)

    # DataLoaders
    loader_train, loader_val = create_data_loaders(args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': []}}

    # Eval
    evaluate(model, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(model, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(model, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(model, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


def train():
    pass


def evaluate():
    pass


def checkpoint():
    pass


def adjust_learning_rate():
    pass


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.device = torch.device("cuda")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)