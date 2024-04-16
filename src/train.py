#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
from torch.utils.data import DataLoader 
import torch

from data import AudioDataset
from solver import Solver
from convolutional_models import Deep_ElectroNet

parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
#parser.add_argument('--segment_len', default=1, type=float,
#                    help='Segment length (seconds)')
#parser.add_argument('--cv_maxlen', default=8, type=float,
#                    help='max audio length (seconds) in cv, to avoid OOM issue.')


# Network architecture
#parser.add_argument('--norm_type', default='gLN', type=str,
#                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
#parser.add_argument('--causal', type=int, default=0,
#                    help='Causal (1) or noncausal(0) training')
#parser.add_argument('--mask_nonlinear', default='relu', type=str,
#                    choices=['relu', 'softmax'], help='non-linear to generate mask')


# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# batch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size')
#parser.add_argument('--num_workers', default=4, type=int,
#                    help='Number of workers to generate minibatch')

# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=2e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=100, type=int,
                    help='Frequency of printing training infomation')


def main(args):
    # Construct Solver
    # data
    print("add BatchNorm and relu")
    tr_dataset = AudioDataset(args.train_dir)
    cv_dataset = AudioDataset(args.valid_dir)
    tr_loader = DataLoader(tr_dataset ,batch_size = 1, shuffle=args.shuffle)
    cv_loader = DataLoader(cv_dataset ,batch_size = 1, shuffle=args.shuffle)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = Deep_ElectroNet()
    print(model)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      betas=(0.5,0.999),
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

