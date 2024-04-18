import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from scipy.io import savemat, wavfile
from pystoi import stoi
from data import AudioDataset
from convolutional_models import LateSupUnet
from CI_vocoder import sound_stimulate

modelpath = 'exp/add_batchnorm_train_r16000_epoch100_half1_norm5_adam_lr2e-5_mmt0_l20_tr/final.pth.tar'
ttdir = 'data/tt'
outdir = 'exp/add_batchnorm_train_r16000_epoch100_half1_norm5_adam_lr2e-5_mmt0_l20_tr/eval'

parser = argparse.ArgumentParser('Generate CI FTM using UNet-CS')
parser.add_argument('--model_path', type=str, required=True, default=modelpath)
parser.add_argument('--test_dir', type=str, default=ttdir)
parser.add_argument('--out_dir', type=str, default=outdir)
parser.add_argument('--use_cuda', type=int, default=1)
parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate of audio file')

def test(args):
    # Load model
    model = LateSupUnet(n_channels=1, bilinear=False)
    model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage)['state_dict'])
    print(model)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    filename = np.load(os.path.join(args.test_dir, 'filenames.npy'))
    test_dataset = AudioDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    os.makedirs(args.out_dir, exist_ok=True)

    avg_test_loss = 0
    loss_weigth = [0.5, 0.5]
    total_stoi = 0
    with torch.no_grad():
        for i, (xt,yftm,clean) in enumerate(test_loader):
            if args.use_cuda:
                xt = xt.cuda()
                yftm = yftm.cuda()
                clean = clean.cuda()
            
            ya = clean.squeeze(0)
            yb = yftm.squeeze(0)

            outftm, denoise = model(xt)
            
            loss = loss_weigth[0] * mse_loss(denoise, ya) + loss_weigth[1] * l1_loss(outftm, yb)
            loss = loss.item()
            print('Loss[{}]: {:.6f}'.format(i + 1, loss))
            avg_test_loss += loss

            estim_sound = sound_stimulate(outftm)
            clean_sound = sound_stimulate(yb)

            stoi_value = stoi(clean_sound, estim_sound, args.sample_rate)
            total_stoi += stoi_value.item()

            print('Average stoi {0:.2f} | Current stoi {1:.2f}'.format(total_stoi / (i + 1), stoi_value.item()))

            savemat(args.out_dir + '/FTM/' + filename[i], {"FTM":outftm})
            wavfile(args.out_dir + 'Sound' + filename[i], args.sample_rate, estim_sound)

    avg_test_loss /= len(test_loader.dataset)
    print(('\n\nAverage Loss: {:.6f}\n\n'.format(avg_test_loss)))

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    test(args)
