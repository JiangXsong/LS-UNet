import os
import numpy as np
from scipy.io import loadmat

# Assuming p.audio_sample_rate = 16000 and p.analysis_rate is defined

# Check for NaN values in ftm and replace with 0.0

def resample(u, ch_stimulate_rate=16000, analysis_rate=900):
    re_sp = []
    num_samples_in = u.shape[1]  # u 是输入信号的数组，假设 u 是二维数组，第二维表示样本数

    # 计算输出信号的样本数
    output_factor =  ch_stimulate_rate/analysis_rate # 假设输出因子为2（示例中使用的输出因子）
    if output_factor == 1:
        return u
    
    num_samples_out = round(num_samples_in * output_factor)

    # 生成输出信号的采样点（sample points）
    sample_points = np.arange(num_samples_out) / output_factor

    # 计算每个采样点对应的采样索引（sample indices）
    sample_indices = np.round(sample_points + 0.5).astype(int)
    sample_indices[0] = 1

    for i in range(sample_indices.shape[0]):

        column = sample_indices[i]  # 获取u的第sample_indices[i]列
        uc = u[:, column-1]
        #print('column = ', uc)
        re_sp.append(uc)
    
    re_sp = np.vstack(re_sp)
    

    return re_sp.T

def sound_stimulate(ftm, analysis_rate=900, audio_sample_rate=16000):
    nan_index = np.isnan(ftm)
    ftm[nan_index] = 0.0

    PS = resample(ftm, ch_stimulate_rate=audio_sample_rate, analysis_rate=analysis_rate)
    
    # Set up the matrix with components for each frequency band using pure sines
    t = np.arange(start=1, stop=PS.shape[1]+1) / audio_sample_rate
    t = t.reshape(1, -1)
    
    char_freqs = loadmat(os.path.join('Vocoder_bins', 'char_freqs.mat'))
    freqs = np.array(char_freqs['char_freqs'])

    sine_component = np.sin(2 * np.pi * np.dot(freqs, t))

    # Reconstruct the sound in each band
    soundbands = sine_component * PS

    # Sum each of the bands together to create the audio signal
    audio = np.sum(soundbands, axis=0)

    return audio

if __name__ == '__main__':
    ftm = np.array(loadmat('Female_List1_1_epoch3.mat')['FTM'])
    ftm[ftm<0] = 0
    audio = sound_stimulate(ftm)
    print("audio: ", audio.shape)
    np.savetxt('audio.txt', audio, fmt='%f')
    #for audio_sp in audio:
    #    print('audio {:.4f}'.format(audio_sp))
