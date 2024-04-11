import argparse
import librosa
import numpy as np
import os
from scipy.io import loadmat

def split_audio(sound_data, length, start=0):
    """
        切成固定長度的音頻檔案 -> 64512 (4.032 s)
    """
    sample_length = sound_data.shape[0]

    if sample_length >= length:
        sample_frame = sound_data[start:length]
    else:
        sample_frame = np.append(sound_data, np.zeros(length-sample_length))
        print("The sound file is below desired length")

    return sample_frame

def spec_to_frame_stack(sp, frame_length, hop_length_frame):
    '''
        分段成同樣長度的frame並堆疊  frame_length = 128
    '''
    sp_length = sp.shape[1]
    sp_list = [sp[:, start:start + frame_length] for start in range(0, sp_length, hop_length_frame)]
    #sound_data_array = np.vstack(sound_data_list)

    return sp_list

def wav_load(file_path, sample_rate):
    '''
        load wav files
    '''
    y, sr = librosa.load(file_path, sr=sample_rate)
    total_duration = librosa.get_duration(y=y, sr=sr) #單位 -> s

    return y, total_duration

def mat_load(file, in_dir):
    '''
        load mat files: {'bandFTM', 'bandFFT'}
    '''
    data = loadmat(os.path.join(in_dir, file))#['bandFTM']['bandFFT']
    FTM_samples = np.array(data['bandFTM'])
    FFT_samples = np.array(data['bandFFT'])
    FFT_samples = np.abs(FFT_samples)
    #length = FTM_samples.shape[1]
    
    return FTM_samples, FFT_samples

def audio_to_numpy(in_dir,  sample_rate, length):
    '''
        資料夾中的音檔轉成固定長度numpy數組
    '''
    in_dir = os.path.abspath(in_dir)
    file_list = os.listdir(in_dir)
    file_list.sort()

    list_sound_array = []

    for file in file_list:                   
        file_path = os.path.join(in_dir, file)
        sp, total_duration = wav_load(file_path, sample_rate)
        nb_sample = total_duration * sample_rate

        if (nb_sample >= length):
            list_sound_array.append(split_audio(sp, length))
        else:
            print(f"The following file {file_path} is below the min duration")
            list_sound_array.append(np.append(sp, np.zeros(length-nb_sample)))

    return np.vstack(list_sound_array)

def clean_file_to_matrix(in_dir, repeat):
    '''
        載入mat檔，並做成training target dataset
    '''
    in_dir = os.path.abspath(in_dir)
    file_list = os.listdir(in_dir)
    file_list.sort()

    list_FTM, list_FTM_array = [], []
        
    for file in file_list:
        FTM_samples, _ = mat_load(file, in_dir)
        list_FTM_array.append(FTM_samples)
        #sp_len = FTM_samples.shape[1]

        #for start in range(0, sp_len-1, frame_length):
        #    list_FTM_array.append(FTM_samples[:, start:start+frame_length])


    if repeat > 0:
        for _ in range(repeat):
            list_FTM.extend(list_FTM_array)
            
    return list_FTM

def audio_to_spec(audio, n_fft=128, hop_length=18):
    '''
        短時傅立葉轉換 STFT
    '''
    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hamming')
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
    #stftaudio_magnitude_db = librosa.amplitude_to_db(stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude, stftaudio_phase

def numpy_audio_to_matrix_spectrogram(numpy_audio, frame_length, n_fft, hop_length_fft):
    """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""

    nb_audio = numpy_audio.shape[0]
    m_mag = [] #[np.zeros((nb_audio, dim_square_spec, frame_length))]
    m_phase = [] #np.zeros((nb_audio, dim_square_spec, frame_length), dtype=complex)

    for i in range(nb_audio):
        mag, phase = audio_to_spec(numpy_audio[i], n_fft, hop_length_fft)
        m_phase.append(phase)
        length = mag.shape[1]
        
        for start in range(0, length-1, frame_length):
            m_mag.append(mag[:, start:start+frame_length])

    return m_mag, m_phase

def audio_to_mel_spec(in_dir, sr=16000, length=64000, n_fft=2048, hop_length=18, n_mels=128):
    '''
        短時傅立葉轉換 STFT -> mel spectrogram
    '''
    in_dir = os.path.abspath(in_dir)
    file_list = os.listdir(in_dir)
    file_list.sort()

    data_list = []

    for file in file_list:
        file_path = os.path.join(in_dir, file)
        y, sr = librosa.load(file_path, sr=sr)
        y = split_audio(y, length)
       
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming', n_mels=n_mels)
        melspec_db = librosa.power_to_db(melspec, ref=np.max) #melspec_db(128, 3585)
        data_list.append(np.array(melspec_db))

        #sp_len = melspec_db.shape[1]
        #for start in range(0, sp_len-1, frame_length):
        #    data_list.append(np.array(melspec_db[:, start:start+frame_length]))

    return data_list