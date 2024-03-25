# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import pickle
import math
import os

import numpy as np
import torch
import torch.utils.data as data

class AudioDataset(data.Dataset):

    def __init__(self, numpy_dir):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        X_in = np.load(os.path.join(numpy_dir, 'noise_Spec.npy'))
        X_out = np.load(os.path.join(numpy_dir, 'clean.npy'))

        data = torch.from_numpy(X_in).float()
        sources = torch.from_numpy(X_out).float()
        
        print(X_in.shape)
        print(X_out.shape)
        
        self.data = data
        self.clean = sources

    def __getitem__(self, index):
        return self.data[index], self.clean[index]

    def __len__(self):
        return len(self.data)

# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

# ------------------------------ utils ------------------------------------
def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = data.DataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
