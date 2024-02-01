#!/usr/bin/env python

import argparse
import pickle
import os
import numpy as np
import librosa
from scipy.io import loadmat

data_dir=""
Out_dir="data"

def FFT_filter(audio, FrameLength, FrameShift, FFT_SIZE): #ACE策略使用的FFT => (audio, 128, 23, 128)
    audio_len = len(audio)
    #overlap_samples = FrameLength-FrameShift     #下一Frame與上一Frame重疊樣本數
    window = np.hamming(FrameLength)
    num_bins = FrameLength//2 + 1

    #初始化
    fftspectrum = []
    yphase = []
    start_sample = 0

    while start_sample < audio_len:
        #分段的結束位置
        end_sample = min(start_sample+FrameLength, audio_len)

        segment = audio[start_sample:end_sample]
        
        #若分段長度不足則補0
        if len(segment) < FrameLength:
            segment = np.append(segment, np.zeros(FrameLength-len(segment)))

        #應用窗函數
        windowed_segment = segment * window

        #進行FFT轉換
        fft_segment = (np.fft.fft(windowed_segment, FFT_SIZE))

        fftspectrum.append(fft_segment[:num_bins])
        yphase.append(np.angle(fft_segment))

        #更新下一Frame起始位置，FrameShift為上一Frame起始位置的位移量
        start_sample += FrameShift              

    fftspectrum = np.array(fftspectrum)
    yphase = np.array(yphase)

    return fftspectrum, yphase

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=16000):
    data = []
    parameter = []
    in_dir = os.path.abspath(in_dir)
    file_list = os.listdir(in_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if out_filename == "clean":
        for mat_file in file_list:
            if not wav_file.endswith('.mat'):
                continue
            mat_path = os.path.join(in_dir, mat_file)       
            samples = loadmat(mat_path)['bandFTM']
            data.append(samples)
        
        with open(os.path.join(out_dir, out_filename + '.txt'), 'wb') as fp:
            pickle.dump(data, fp)# indent=4
    else:
        for wav_file in file_list:
            if not wav_file.endswith('.wav'):
                continue
            wav_path = os.path.join(in_dir, wav_file)       
            samples, _ = librosa.load(wav_path, sr=sample_rate)
            sp, yphase = FFT_filter(samples, 128, 18, 128)
            data.append(sp)
            parameter.append([yphase, wav_file.replace(file_list, '')])

        with open(os.path.join(out_dir, out_filename + '_parameter.txt'), 'wb') as fp:
            pickle.dump(parameter, fp)# indent=4
        with open(os.path.join(out_dir, out_filename + '.txt'), 'wb') as fp:
            pickle.dump(data, fp)# indent=4

def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['-2dB', '-6dB', '2dB', "6dB", 'clean']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("data preprocessing")
    parser.add_argument('--in-dir', type=str, default=data_dir,
                        help='Directory path of audio data including tr, cv and tt')
    parser.add_argument('--out-dir', type=str, default=Out_dir,
                        help='Directory path to put output files')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    print(args)
    preprocess(args)
