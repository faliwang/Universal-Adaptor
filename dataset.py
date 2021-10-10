import os
import json

import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from extract import Extractor
from itertools import product


class AudioDataset(Dataset):
    def __init__(self, data_dir, config_dir, fix_config_dir, segment_len=128):
        self.data_dir = data_dir
        self.files = self.get_wav_npy_files(config_dir)
        self.src_cfgs = {
            idx:cfg for cfg, idx in get_configs(config_dir+'/config')}
        self.src_exts = {
            idx:Extractor(self.src_cfgs[idx]) for idx in self.src_cfgs}
        self.tgt_cfgs = [generate_config(fix_config_dir) for _ in range(100)]
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
        mid_wav, tgt_wav = self.trim_wav(mid_wav, tgt_wav, seg_l)

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

    def trim_wav(self, mid, tgt, seg_l):
        l = min(len(mid), len(tgt))
        mid, tgt = mid[:l], tgt[:l]
        if l > seg_l:
            s = np.random.randint(0, l-seg_l)
            mid, tgt = mid[s:s+seg_l], tgt[s:s+seg_l]
        elif l < seg_l:
            mid = np.pad(mid, (0, seg_l-l), 'constant', constant_values = (0, 0))
            tgt = np.pad(tgt, (0, seg_l-l), 'constant', constant_values = (0, 0))
        return mid, tgt


class InferenceDataset(Dataset):
    def __init__(self, input_dir, config_dir):
        data = []
        models = os.listdir(input_dir)
        for model_name in models:
            model_dir = os.path.join(input_dir, model_name)
            inputs = os.listdir(model_dir)
            inputs.sort()
            for input_name in inputs:
                input_path = os.path.join(model_dir, input_name)
                data.append((input_path, model_name, input_name))
        self.data = data

        self.configs = get_configs(config_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_path, model_name, input_name = self.data[index]
        mel = torch.from_numpy(np.load(input_path)).float()
        tgt_config = ''
        for config, config_name in self.configs:
            if config_name == model_name.split('_')[-1]:
                tgt_config = config
                break
        tgt_cfg_list = convert_config(tgt_config)
        return torch.FloatTensor(mel), torch.FloatTensor(tgt_cfg_list), model_name, input_name


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


def generate_config(config_dir):
    config = {}
    config['wav_config'] = generate_wav_config()
    config['spec_config'] = generate_spec_config()
    config['post_config'] = generate_post_config()
    config['github_repo'] = None
    
    while check_same_config(config, config_dir):
        config['spec_config'] = generate_spec_config()

    return config


def generate_wav_config():
    peak_norm = random.uniform(0.9, 1.0)
    wav_config = {"sample_rate": 22050, "normalize_loudness": None, "peak_norm": peak_norm, "highpass_cutoff": 0.0}
    trim_not = {"trim_silence": False, "trim_silence_threshold_in_db": 0, "trim_frame_size": 0, "trim_hop_size": 0}
    wav_config.update(trim_not)
    return wav_config


def generate_spec_config():
    win_length = random.choice([800, 900, 1024, 1100, 1200])
    n_fft = max(win_length, random.choice([1024, 2048]))
    center = random.choice([True, False])
    pad = 0 if center else (n_fft-win_length//4)//2
    fmin = random.choices([0, 30, 50, 70, 90, 125])[0]
    fmax = random.choices([7600, 8000, 9500, 11025])[0]
    spec_config = {
        "preemphasis": None,
        "n_fft": n_fft, "hop_length": win_length//4, "win_length": win_length, "window": "hann",
        "left_pad": pad, "right_pad": pad, "pad_mode": "reflect",  "center": center,
        "stft_power": 1,
        "mel_spec": True,
        "num_mels": 80, "fmin": fmin, "fmax": fmax
    }
    return spec_config


def generate_post_config():
    log_base = random.choice([10, 'e'])
    log_factor = random.choice([20, 1])\
        if log_base == 10 else 1
    normalize_spec = log_factor == 20
    ref_level_db = random.choice([20, 0]) \
        if normalize_spec else 0
    post_config = {
        "amp_to_db": True,
        "log_base": log_base,
        "log_factor": log_factor,
        "normalize_spec": normalize_spec,
        "ref_level_db": ref_level_db,
        "min_level_db": -100
    }
    return post_config


def check_same_config(config, cfg_dir):
    for cfg_path in os.listdir(cfg_dir):
        with open(os.path.join(cfg_dir, cfg_path), 'r') as f:
            cfg = json.load(f)
        if (cfg['spec_config'] == config['spec_config']):
            return 1
    return 0


def convert_config(config):
    wav_cfg = config["wav_config"]
    wav_cfg = list(sorted(wav_cfg.items()))
    spec_cfg = config["spec_config"]
    spec_cfg = list(sorted(spec_cfg.items()))
    to_log = [
        "fmin", "fmax", "n_fft", "left_pad", "right_pad",
        "hop_length", "win_length"]
    cfg_list = []
    for config in [wav_cfg, spec_cfg]:
        for param_name, param in config:
            if param_name == 'peak_norm':
                cfg_list.append(param)
            elif param_name in to_log:
                cfg_list.append(np.log1p(param))
            
    return cfg_list