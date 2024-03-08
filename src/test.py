import argparse
import os
import torch
#import torch.nn as nn

from torch.utils.data import DataLoader
from scipy.io import savemat
from data import AudioDataset
from convolutional_models import LateSupUnet

parser = argparse.ArgumentParser('Generate CI FTM using UNet-CS')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--test_dir', type=str, default=None)
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--use_cuda', type=int, default=0)

def test(args):
    # Load model
    model = LateSupUnet(n_channels=1, bilinear=False)
    model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
    print(model)
    #criterion = nn.MSELoss()
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    test_dataset = AudioDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    os.makedirs(args.out_dir, exist_ok=True)
    ftm_list = []
    
    with torch.no_grad():
        for i, (xt,yt) in enumerate(test_loader):
            if args.use_cuda:
                xt = xt.cuda()
                yt = yt.cuda()
            
            xa = xt.unsqueeze(0)
            target_ftm = yt.squeeze(0)

            estimate_ftm = model(xa)
            ftm_list.append(estimate_ftm)
            #loss = criterion(estimate_ftm - target_ftm)
