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
from itertools import permutations


class AudioDataset(Dataset):
    def __init__(self, data_dir, data_type, config_dir, segment_len=128):
        self.data_type = data_type
        if data_type == 'wav':
            self.files = self.get_wav_files(data_dir)
        elif data_type == 'npy':
            self.files = self.get_mel_files(data_dir)
        else:
            raise NotImplementedError(f'we only implement data type wav & npy, but we got {data_type}')
        
        self.configs = get_configs(config_dir)
        self.extractors = [Extractor(config) for config, _ in self.configs]
        perms = permutations(range(len(self.configs)), 2)
        self.perms = list(perms)
        self.segment_len = segment_len
 

    def __len__(self):
        return len(self.files)
 

    def __getitem__(self, index):
        if self.data_type == 'wav':
            wav_path = self.files[index]

            perm_index = index % len(self.perms)
            src, tgt = self.perms[perm_index]
            src_ext = self.extractors[src]
            tgt_ext = self.extractors[tgt]
            tgt_config, _ = self.configs[tgt]

            # getting input mel
            src_wav = src_ext.load(wav_path)
            src_mel = src_ext.convert(src_wav)
            mid_wav = src_ext.inverse(src_mel)
            tgt_mel_init = tgt_ext.convert(mid_wav)
            input = torch.from_numpy(tgt_mel_init)
            
            # getting ground truth mels
            tgt_wav = tgt_ext.load(wav_path)
            tgt_mel_gt = tgt_ext.convert(tgt_wav)
            gt = torch.from_numpy(tgt_mel_gt)
        
        else:
            input_path, gt_path, model_name = self.files[index]
            model_name = model_name.split('_')[-1]
            for config, config_name in self.configs:
                if config_name == model_name:
                    tgt_config = config
                    break
            input = torch.from_numpy(np.load(input_path))
            gt = torch.from_numpy(np.load(gt_path))

        input, gt = self.trim_mels(input, gt)
        tgt_cfg_list = convert_config(tgt_config)

        # print(torch.FloatTensor(input))
        # print(torch.FloatTensor(gt))
        # print(torch.FloatTensor(tgt_cfg_list))
        return torch.FloatTensor(input), torch.FloatTensor(gt), torch.FloatTensor(tgt_cfg_list)
    

    def get_wav_files(self, wav_dir):
        """
            return list: [wav_path]
        """
        wav_files = os.listdir(wav_dir)
        files = []
        for f_path in wav_files:
            if f_path.endswith('.wav'):
                files.append(os.path.join(wav_dir, f_path))
        assert len(files) != 0, 'Num of wav in wav dir should not be zero'
        return sorted(files) 

    
    def get_mel_files(self, mel_dir):
        """
            return list: [(input_path, gt_path, model_name)]
        """
        gt_dir = os.path.join(mel_dir, 'ground_truth')
        input_dir = os.path.join(mel_dir, 'input')
        assert os.path.exists(gt_dir), 'There is no ground truth directory in npy dir'
        assert os.path.exists(input_dir), 'There is no input directory in npy dir'
        input_models = sorted(os.listdir(input_dir))
        gt_models = sorted(os.listdir(gt_dir))
        files = []
        for model_name in input_models:
            if model_name.split('_')[-1] not in gt_models:
                raise NameError(f'the models in input should have ground truth, but we get {model_name} which is not in {gt_models}')
            gt_model = model_name.split('_')[-1]
            input_model_dir = os.path.join(input_dir, model_name)
            gt_model_dir = os.path.join(gt_dir, gt_model)
            input_mels = sorted(os.listdir(input_model_dir))
            gt_mels = sorted(os.listdir(gt_model_dir))
            for input_mel, gt_mel in zip(input_mels, gt_mels):
                if input_mel != gt_mel:
                    print("we get wrong pair of files: input {} & ground truth {} in {}".format(input_mel, gt_mel, model_name))
                    continue
                input_path = os.path.join(input_model_dir, input_mel)
                gt_path = os.path.join(gt_model_dir, gt_mel)
                files.append((input_path, gt_path, model_name))
        return files


    def trim_mels(self, input, gt):
        input_length = input.shape[-1]
        gt_length = gt.shape[-1]
        
        # Segmemt mel-spectrogram into "segment_len" frames.
        if input_length > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, input_length - self.segment_len)
            # Get a segment with "segment_len" frames.
            input = input[:, start:start+self.segment_len]
            if gt_length >= start+self.segment_len:
                gt = gt[:, start:start+self.segment_len]
            else:
                gt = gt[:, start:]
        elif input_length < self.segment_len:
            input = F.pad(input, (0, self.segment_len - input_length))

        if gt.shape[-1] > self.segment_len:
            gt = gt[:, :self.segment_len]
        elif gt.shape[-1] < self.segment_len:
            gt = F.pad(gt, (0, self.segment_len - gt.shape[-1]))    
        assert gt.shape[-1] == self.segment_len, "gt length {} != segment_len {}".format(gt.shape[-1], self.segment_len)
        assert input.shape[-1] == self.segment_len, "input length {} != segment_len {}".format(input.shape[-1], self.segment_len)

        return input, gt


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