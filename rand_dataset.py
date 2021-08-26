import os
import json

from numpy.core.fromnumeric import sort
import torch
import math
import random
import collections
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from extract import Extractor
from itertools import product
from rand_prep import rand_config


class AudioDataset(Dataset):
    def __init__(self, data_dir, data_type, config_dir, segment_len=128):
        self.data_dir = data_dir
        self.files = self.get_wav_npy_files(config_dir)
        self.src_cfgs = {
            idx:cfg for cfg, idx in get_configs(config_dir+'/config')}
        self.src_exts = {
            idx:Extractor(self.src_cfgs[idx]) for idx in self.src_cfgs}
        self.tgt_cfgs = [rand_config() for _ in range(len(self.src_cfgs))]
        self.tgt_exts = [Extractor(cfg) for cfg in self.tgt_cfgs]
        self.segment_len = segment_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wav_path, _ = self.files[index]
        r = random.randint(0, len(self.tgt_cfgs)-1)
        tgt_cfg = self.tgt_cfgs[r]
        tgt_ext = self.tgt_exts[r]

        # get mid & tgt wav
        mid_wav = np.load(wav_path)
        bn = os.path.basename(wav_path).split('.')[0]
        tgt_wav = tgt_ext.load(
            os.path.join(self.data_dir, bn+'.wav'))
        seg_l = (self.segment_len+1)*\
            tgt_cfg['spec_config']['hop_length']
        mid_wav = self.trim_wav(mid_wav, seg_l)
        tgt_wav = self.trim_wav(tgt_wav, seg_l)

        # gen inp & tgt mel
        inp_mel = tgt_ext.convert(mid_wav)[:, :self.segment_len]
        tgt_mel = tgt_ext.convert(tgt_wav)[:, :self.segment_len]

        inp = torch.from_numpy(inp_mel)
        tgt = torch.from_numpy(tgt_mel)
        tgt_cfg_list = convert_config(tgt_cfg)

        return torch.FloatTensor(inp), torch.FloatTensor(tgt), torch.FloatTensor(tgt_cfg_list)
    
    def get_wav_npy_files(self, wav_npy_dir):
        files = []
        for idx in range(len(os.listdir(wav_npy_dir))-1):
            subdir = os.path.join(wav_npy_dir, str(idx))
            for f_path in os.listdir(subdir):
                if f_path.endswith('.npy'):
                    files.append([os.path.join(subdir, f_path), str(idx)])
        return files

    def trim_wav(self, wav, seg_l):
        if wav.shape[0] > seg_l:
            start = np.random.randint(0, len(wav)-seg_l)
            wav = wav[start:start+seg_l]
        elif wav.shape[0] < seg_l:
            wav = np.pad(wav, (0, seg_l-wav.shape[0]), 'constant', constant_values = (0, 0))
        return wav


def get_configs(config_dir):
    """
        return list: [(config(dict), config_name)]
    """
    config_dir = os.path.abspath(config_dir)
    config_paths = os.listdir(config_dir)
    config_paths.sort()
    configs = []
    for config_path in config_paths:
        config_name = config_path[:-5].split('_')[-1]
        with open(os.path.join(config_dir, config_path), 'r') as f:
            config = json.load(f)
            configs.append((config, config_name))
    return configs


def convert_config(config):
    wav_cfg = config["wav_config"]
    wav_cfg = list(sorted(wav_cfg.items()))
    spec_cfg = config["spec_config"]
    spec_cfg = list(sorted(spec_cfg.items()))
    post_cfg = config["post_config"]
    post_cfg = list(sorted(post_cfg.items()))
    cfg_list = []
    window_dict = {'hann': 1, None: 0}
    pad_mode_dict = {'reflect': 1, None: 0}
    for config in [wav_cfg, spec_cfg, post_cfg]:
        for param_name, param in config:
            if param_name == 'window':
                cfg_list.append(window_dict[param])
            elif param_name == 'pad_mode':
                cfg_list.append(pad_mode_dict[param])
            elif param == True:
                cfg_list.append(1)
            elif param == None or param == False:
                cfg_list.append(0)
            elif param == "e":
                cfg_list.append(math.exp(1))
            else:
                if type(param) == int or type(param) == float:
                    cfg_list.append(param)
                else:
                    raise ValueError(f"We got unknown parameter in config: {param_name}: {param}")
    return cfg_list