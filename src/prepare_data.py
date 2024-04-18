#!/usr/bin/env python

import argparse
import os
import numpy as np
from data_tools import clean_file_to_matrix, audio_to_mel_spec

data_dir = "/home/song/LS-UNet/"
Out_dir = "data"

def prepare_one_dir(data_type, in_dir, out_dir, snr_list, sample_rate, n_fft):
  noise_data_list, list_clean = [], []
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  for snr in snr_list: 
    noise_data_list.extend(audio_to_mel_spec(os.path.join(in_dir, data_type, snr),
                                             sr=sample_rate,
                                             length=64000,
                                             n_fft=n_fft,
                                             hop_length=18,
                                             n_mels=128))  

  print("sp_noise ", len(noise_data_list))  
  np.save(out_dir + '/noise_Spec', noise_data_list)    
  #np.save(out_dir + '/noise_Phase', yphase)

  clean_path = os.path.join(in_dir, data_type, 'clean')
  if os.path.exists(clean_path):
    repeat = len(snr_list) 
    FTM_clean = clean_file_to_matrix(os.path.join(in_dir, data_type, 'clean'), 
                                     repeat)
    print("FTM_clean ", len(FTM_clean))
    np.save(out_dir + '/FTM_clean', FTM_clean)

  clean = audio_to_mel_spec(os.path.join(in_dir, data_type, 'clean_wav'),
                                         sr=sample_rate,
                                         length=64000,
                                         n_fft=n_fft,
                                         hop_length=512,
                                         n_mels=128)
  for _ in range(repeat):
    list_clean.extend(clean)
  print("sp_noise ", len(list_clean))
  np.save(out_dir + '/clean', list_clean)
    

def prepare_data(args):
  for data_type in ['tr', 'cv', 'tt']:
    out_dir = os.path.join(args.out_dir, data_type)

    if data_type == 'tr':
      snr_list = ['-10.0', '-8.0', '-7.0', '-5.0', '-4.0', '-2.0', '-1.0', '0', '1.0', '2.0', '4.0', '5.0', '7.0', '8.0', '10.0']
      prepare_one_dir(data_type, args.in_dir, out_dir, snr_list, args.sample_rate, args.n_fft)

    elif data_type == 'cv':
      snr_list = ['-3.0', '-6.0', '-9.0', '3.0', '6.0', '9.0']
      prepare_one_dir(data_type, args.in_dir, out_dir, snr_list, args.sample_rate, args.n_fft)

    elif data_type == 'tt':
      snr_list = ['-5dB', '5dB', '0dB']
      files_save_ls = []
      save_ls = os.listdir(os.path.abspath(os.path.join(args.in_dir, data_type, 'clean_wav')))
      save_ls.sort()
      for _ in range(len(snr_list)):
        files_save_ls.extend(save_ls)
      np.save(out_dir + '/filenames', files_save_ls)

      prepare_one_dir(data_type, args.in_dir, out_dir, snr_list, args.sample_rate, args.n_fft)

    else:
      print('Please checkout your filename! ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("data preprocessing")
    parser.add_argument('--in_dir', type=str, default=data_dir,
                        help='Directory path of audio data including tr, cv and tt')
    parser.add_argument('--out_dir', type=str, default=Out_dir,
                        help='Directory path to put output files')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate of audio file')
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--frame_length', type=int, default=256) #4(s)*16000
    args = parser.parse_args()
    print(args)
    prepare_data(args)
