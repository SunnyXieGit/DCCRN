from torch.utils.data import Dataset, DataLoader
import librosa as lib
import os
import numpy as np
import torch

#加载WAV文件并将其分割成指定持续时间的帧。
def load_wav(path, frame_dur, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    win = int(frame_dur / 1000 * sr)
    #return torch.tensor(np.split(signal, int(len(signal) / win), axis=0))
    return torch.tensor(np.array(np.split(signal, int(len(signal) / win), axis=0)))

    # 确保信号为float32类型
    #signal = signal.astype(np.float32)
    # 计算每个帧的采样点数
    #win = int(frame_dur / 1000 * sr)
    # 将信号分割成帧，并转换为PyTorch张量，明确指定数据类型
    #frames = np.split(signal, int(len(signal) / win), axis=0)
    #return torch.tensor(frames, dtype=torch.float32)


#自定义数据集类，用于加载成对的噪声和干净音频文件。
class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, loader=load_wav, frame_dur=37.5):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.loader = loader
        self.frame_dur = frame_dur

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        return self.loader(noisy_file, self.frame_dur), self.loader(clean_file, self.frame_dur)

    def __len__(self):
        return len(self.noisy_paths)

#载WAV文件并将其分割成指定持续时间和跳跃长度的帧。
def load_hop_wav(path, frame_dur, hop_dur, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    win = int(frame_dur / 1000 * sr)
    hop = int(hop_dur / 1000 * sr)
    rest = (len(signal) - win) % hop
    signal = np.pad(signal, (0, hop - rest), "constant")
    n_frames = int((len(signal) - win) // hop)
    strides = signal.itemsize * np.array([hop, 1])
    return torch.tensor(np.lib.stride_tricks.as_strided(signal, shape=(n_frames, win), strides=strides))

#自定义数据集类，用于加载成对的噪声和干净音频文件，并支持重叠帧
class WavHopDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, frame_dur, hop_dur, loader=load_hop_wav):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.loader = loader
        self.frame_dur = frame_dur
        self.hop_dur = hop_dur

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        return self.loader(noisy_file, self.frame_dur, self.hop_dur), \
               self.loader(clean_file, self.frame_dur, self.hop_dur)

    def __len__(self):
        return len(self.noisy_paths)
