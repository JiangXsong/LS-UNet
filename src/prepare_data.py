#!/usr/bin/env python

import argparse
import os
import numpy as np
from data_tools import audio_to_numpy, clean_file_to_matrix, numpy_audio_to_matrix_spectrogram

def prepare_data(args):
  for data_type in ['tr', 'cv', 'tt']:
    if data_type == 'tr':
      snr_list = ['-2dB', '-6dB', '2dB', "6dB"]
      noise_data_list = []

      for snr in snr_list:
        noise_data_list.append(audio_to_numpy(os.path.join(args.in_dir, data_type, snr),
                                              sample_rate=args.sample_rate,
                                              frame_length=args.frame_length))
      
      noise_data_array = np.vstack(noise_data_list)
      dim_square_spec = int(args.n_fft/2)+1

      sp_noise, yphase = numpy_audio_to_matrix_spectrogram(noise_data_array, dim_square_spec, args.n_fft, hop_length_fft=18)

      sp_clean = clean_file_to_matrix(os.path.join(args.in_dir, data_type, 'clean'),
                          len(snr_list),
                          dim_square_spec)
      
      np.save(args.out_dir + 'tr_noise_Spec', sp_noise)
      np.save(args.out_dir + 'tr_clean', sp_clean)

      np.save(args.out_dir + 'tr_noise_Phase', yphase)

    elif data_type == 'cv':
      snr_list = ['-2dB', '-6dB', '2dB', "6dB"]
      noise_data_list = []

      for snr in snr_list:
        noise_data_list.append(audio_to_numpy(os.path.join(args.in_dir, data_type, snr),
                                              sample_rate=args.sample_rate,
                                              frame_length=args.frame_length))
      
      noise_data_array = np.vstack(noise_data_list)
      dim_square_spec = int(args.n_fft/2)+1

      sp_noise, yphase = numpy_audio_to_matrix_spectrogram(noise_data_array, dim_square_spec, args.n_fft, hop_length_fft=18)

      sp_clean = clean_file_to_matrix(os.path.join(args.in_dir, data_type, 'clean'),
                          len(snr_list),
                          dim_square_spec)
      
      np.save(args.out_dir + 'cv_noise_Spec', sp_noise)
      np.save(args.out_dir + 'cv_clean', sp_clean)

      np.save(args.out_dir + 'cv_noise_Phase', yphase)

    else:
      print('Please checkout your filename! ')