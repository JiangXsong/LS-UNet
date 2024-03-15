import argparse
import librosa
import numpy as np
import os
from scipy.io import loadmat

def split_audio(sound_data, frame_length, start=0):
    """
        切成固定長度的音頻檔案
    """
    sample_length = sound_data.shape[0]

    if sample_length >= frame_length:
        sample_frame = sound_data[start:frame_length]
    else:
        sample_frame = sound_data
        print("The sound file is below the desire length")

    return sample_frame

def audio_frame_stack(sound_data, frame_length, hop_length_frame):
    '''
        分段成同樣長度的frame並堆疊
    '''
    sequence_sample_length = len(sound_data)
    sound_data_list = [sound_data[start:start + frame_length] for start in range(0, sequence_sample_length-frame_length+1, hop_length_frame)]
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array

def wav_load(file_path, sample_rate):
    '''
        load wav files
    '''
    y, sr = librosa.load(file_path, sr=sample_rate)
    total_duration = librosa.get_duration(y=y, sr=sr) #單位 -> s

    return y, total_duration

def mat_load(file, in_dir):
    '''
        load mat files
    '''
    samples = loadmat(os.path.join(in_dir, file))['bandFTM']
    samples = np.array(samples)
    length = samples.shape[1]
    
    return samples, length

def audio_to_numpy(in_dir,  sample_rate, frame_length, hop=False):
    in_dir = os.path.abspath(in_dir)
    file_list = os.listdir(in_dir)
    file_list.sort()

    list_sound_array = []
    if not hop:
        hop_length_frame = frame_length

    for file in file_list:                   
        #資料夾中的音檔轉成numpy
        file_path = os.path.join(in_dir, file)
        sp, total_duration = wav_load(file_path, sample_rate)
        nb_sample = total_duration * sample_rate

        if (nb_sample >= frame_length):
            list_sound_array.append(audio_frame_stack(sp, frame_length, hop_length_frame))
        else:
            print(
                f"The following file {file_path} is below the min duration")

    return np.vstack(list_sound_array)

def clean_file_to_matrix(in_dir, frame_length, repeat, dim_square_spec):
    '''
        載入mat檔，並做成training target dataset
    '''
    in_dir = os.path.abspath(in_dir)
    file_list = os.listdir(in_dir)
    file_list.sort()

    list_target_array, list_target = [], []
        
    for file in file_list:
        samples, length = mat_load(file, in_dir)
        for i in range(0, length, frame_length):
            list_target_array.append(samples[:, i:i+frame_length])

    if repeat > 0:
        for _ in range(repeat):
            list_target.extend(list_target_array)
    
    numpy_target = np.vstack(list_target)
    nb_target = numpy_target.shape[0]

    n_ftm = np.zeros((nb_target, frame_length, dim_square_spec))

    for i in range(nb_target):
        n_ftm[i, :, :] = numpy_target[i]
    
    return n_ftm

def audio_to_spec(audio, n_fft=128, hop_length=18):
    '''
        短時傅立葉轉換 STFT
    '''
    stftaudio = librosa.stft(audio, n_fft, hop_length, window='hamming')
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
    #stftaudio_magnitude_db = librosa.amplitude_to_db(stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude, stftaudio_phase

def numpy_audio_to_matrix_spectrogram(numpy_audio, frame_length, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""

    nb_audio = numpy_audio.shape[0]
    m_mag = np.zeros((nb_audio, frame_length, dim_square_spec))
    m_phase = np.zeros((nb_audio, frame_length, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag[i, :, :], m_phase[i, :, :] = audio_to_spec(numpy_audio[i], n_fft, hop_length_fft)

    return m_mag, m_phase