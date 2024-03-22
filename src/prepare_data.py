#!/usr/bin/env python

import argparse
import os
import numpy as np
from data_tools import audio_to_numpy, clean_file_to_matrix, numpy_audio_to_matrix_spectrogram

data_dir = "/home/song/LS-UNet/"
Out_dir = "data"

def prepare_one_dir(data_type, in_dir, out_dir, snr_list, sample_rate, frame_length, n_fft):
  noise_data_list = []
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  for snr in snr_list:      
    noise_data_list.append(audio_to_numpy(os.path.join(in_dir, data_type, snr),
                                          sample_rate=sample_rate,
                                          length=64512))    
  noise_data_array = np.vstack(noise_data_list)
  #dim_square_spec = int(n_fft/2)+1

  sp_noise, yphase = numpy_audio_to_matrix_spectrogram(noise_data_array,
                                                       frame_length,
                                                       n_fft,
                                                       hop_length_fft=18)
    
  np.save(out_dir + '/noise_Spec', sp_noise)    
  np.save(out_dir + '/noise_Phase', yphase)

  clean_path = os.path.join(in_dir, data_type, 'clean')
  if os.path.exists(clean_path):
    repeat = len(snr_list) - 1
    sp_clean = clean_file_to_matrix(os.path.join(in_dir, data_type, 'clean'), frame_length, repeat)
    np.save(out_dir + '/clean', sp_clean)

def prepare_data(args):
  for data_type in ['tr', 'cv', 'tt']:
    out_dir = os.path.join(args.out_dir, data_type)

    if data_type == 'tr':
      snr_list = ['-2dB', '-6dB', '2dB', "6dB"]
      prepare_one_dir(data_type, args.in_dir, out_dir, snr_list, args.sample_rate, args.frame_length, args.n_fft)

    elif data_type == 'cv':
      snr_list = ['-5dB']
      prepare_one_dir(data_type, args.in_dir, out_dir, snr_list, args.sample_rate, args.frame_length, args.n_fft)

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
    parser.add_argument('--n_fft', type=int, default=128)
    parser.add_argument('--frame_length', type=int, default=128) #4(s)*16000
    args = parser.parse_args()
    print(args)
    prepare_data(args)