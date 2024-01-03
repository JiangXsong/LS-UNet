import argparse
import json
import os
import numpy as np
import librosa

data_dir=""
Out_dir="data"

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=16000):
    file_infos = []
    data = []
    parameter = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)       
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        sp, yphase = Spectrum(samples, 512, 256, 512, 2)
        data.append(sp)
        parameter.append([yphase, wav_file.replace(wav_list, '')])
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['noisy_03', 'noisy_06', 'noisy_09', 'clean']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)

def Spectrum(sig, FrameLength, FrameRate, FFT_SIZE, flag):       #(sig, 512, 256, 512, 2)
    Len = len(sig)
    ncols = int((Len-FrameLength)/FrameRate)
    fftspectrum = np.zeros([FFT_SIZE, ncols])
    Spectrum = np.zeros([FFT_SIZE//2+1, ncols])
    En = np.zeros([1, ncols])
    wind = np.hamming(FrameLength)
    
    x_seg = []
    fftspectrum = []
    yphase = []
    Spec = []
    i = 0
    for t in range(0, Len-FrameLength, FrameRate):
        #pdb.set_trace()
        x_seg.append(wind*(sig[t:(t+FrameLength)]))
        fftspectrum.append(np.fft.fft(x_seg[i], FFT_SIZE))
        yphase.append(np.angle(fftspectrum[i]))
        Spec.append(np.abs(fftspectrum[i][0:FFT_SIZE//2+1]))
        i += 1
    
    fftspectrum = np.array(fftspectrum)
    yphase = np.array(yphase)
    Spec = np.array(Spec)
    #pdb.set_trace()
    if flag==2:
        Spec = Spec**2
    elif flag==1:
        Spec = Spec
    else:
        Spec = fftspectrum[0:FFT_SIZE//2, :]
    return np.log10(Spec, where=Spec>0), yphase

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
