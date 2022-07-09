import os
import json
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from extract import Extractor
from multiprocessing import Pool, cpu_count
from utils import audio


def interpolate(mel, src_ext, tgt_ext):
    src_hop_length = src_ext.spec_config["hop_length"]
    tgt_hop_length = tgt_ext.spec_config["hop_length"]
    src_sample_rate = src_ext.wav_config["sample_rate"]
    tgt_sample_rate = tgt_ext.wav_config["sample_rate"]
    scale_factor = ( tgt_sample_rate / tgt_hop_length ) / ( src_sample_rate / src_hop_length )
    mel = torch.Tensor(mel)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    mel = F.interpolate(mel, scale_factor=scale_factor, recompute_scale_factor=True)
    mel = mel.squeeze(0).numpy()
    return mel


def process(x):
    src_ext, tgt_ext, n_iter, fpath, outdir = x
    S = np.load(fpath)
    S = src_ext.post_inverse(S)
    S = interpolate(S, src_ext, tgt_ext)
    if tgt_ext.post_config["amp_to_db"]:
        S = audio.amp_to_db(S, {"log_base": 'e', "log_factor": 1})
    S = tgt_ext.post_convert(S)
    idx = fpath.split('/')[-1].split('.')[0]
    outpath = os.path.join(outdir, f'{idx}.npy')
    np.save(outpath, S, allow_pickle=False)


def transform(src_config, tgt_config, data, extension,
             n_iter, n_workers, outdir):
    # Load config from json
    src_cfgs = []
    if src_config == 'all':
        src_configs = os.listdir('./config')
        src_configs.sort()
        for src_config in src_configs:
            src_config_json = os.path.join(os.getcwd(), 'config', src_config)
            with open(src_config_json, 'r') as f:
                src_cfg = json.load(f)
                src_cfgs.append(src_cfg)
    else:
        with open(src_config, 'r') as f:
            src_cfg = json.load(f)
            src_cfgs.append(src_cfg)
            
    tgt_cfgs = []
    if tgt_config == 'all':
        tgt_configs = os.listdir('./config')
        tgt_configs.sort()
        for tgt_config in tgt_configs:
            tgt_config_json = os.path.join(os.getcwd(), 'config', tgt_config)
            with open(tgt_config_json, 'r') as f:
                tgt_cfg = json.load(f)
                tgt_cfgs.append(tgt_cfg)
    else:
        with open(tgt_config, 'r') as f:
            tgt_cfg = json.load(f)
            tgt_cfgs.append(tgt_cfg)

    # Build Extractor
    src_exts = []
    for src_cfg in src_cfgs:
        src_ext = Extractor(src_cfg)
        src_exts.append(src_ext)
    tgt_exts = []
    for tgt_cfg in tgt_cfgs:
        tgt_ext = Extractor(tgt_cfg)
        tgt_exts.append(tgt_ext)

    # Search for files
    data_list = [x for x in os.listdir(data) if x.endswith(extension)]
    if len(data_list) == 0:
        config_name = src_configs[0][:-5].split('_')[1]
        data_list = [x for x in os.listdir(os.path.join(data, config_name)) if x.endswith(extension)]
    print(f'{len(data_list)} {extension[1:]} files found in {data}')
    f_list = []
    for i in range(len(src_configs)):
        src_config_name = src_configs[i][:-5].split('_')[1]
        for j in range(len(tgt_configs)):
            tgt_config_name = tgt_configs[j][:-5].split('_')[1]
            out_dir = os.path.join(outdir, src_config_name + '_' + tgt_config_name)
            os.makedirs(out_dir, exist_ok=True)
            _list = [(src_exts[i], tgt_exts[j], n_iter, os.path.join(data, src_config_name, x), out_dir) for x in data_list]
            f_list += _list    

    # Extract
    if len(f_list) != 0:
        n_workers = min(max(1, n_workers), cpu_count())
        with Pool(processes=n_workers) as pool:
            for _ in pool.imap_unordered(process, tqdm(f_list)):
                pass
        print('Completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess for TTS and vocoder')
    parser.add_argument('--src_config', '-sc', metavar='FILE',
                        default='all', help='The src_config file of source features')
    parser.add_argument('--tgt_config', '-tc', metavar='FILE',
                        default='all', help='The src_config file of target features')
    parser.add_argument('--data', '-d', metavar='DATA',
                        default='data', help='The dataset folder')
    parser.add_argument('--extension', '-e', metavar='EXT', default='.npy',
                        help='File extension to search for in dataset folder')
    parser.add_argument('--n_iter', '-ni', metavar='N', type=int, default=32,
                        help='The number of iterations for Griffin-Lim')
    parser.add_argument('--n_workers', '-nw', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    parser.add_argument('--outdir', '-o', metavar='OUT', default='res',
                        help='Output directory')
    args = parser.parse_args()
    transform(**vars(args))
