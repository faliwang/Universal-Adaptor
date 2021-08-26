import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from extract import Extractor
from multiprocessing import Pool, cpu_count


def rand_config():
    trim_p = 0.2
    fmin_p = 0.4
    fmax_p = 0.4
    
    # generate wav config
    peak_norm = random.uniform(0.9, 1.0)
    wav_config = {"sample_rate": 22050, "normalize_loudness": None, "peak_norm": peak_norm}
    trim = {"trim_silence": True, "trim_silence_threshold_in_db": 60, "trim_frame_size": 2048, "trim_hop_size": 512}
    trim_not = {"trim_silence": False, "trim_silence_threshold_in_db": 0, "trim_frame_size": 0, "trim_hop_size": 0}
    trim_silence = random.choices([trim, trim_not], weights=[trim_p, 1-trim_p])[0]
    wav_config.update(trim_silence)

    # generate spec config
    n_fft = random.choice([512, 1024, 2048])
    center = random.choice([True, False])
    pad = 0 if center else n_fft * 3 // 8
    fmin = random.choices([0, 20, 40], weights=[fmin_p, 1-2*fmin_p, fmin_p])[0]
    fmax = random.choices([8000, None], weights=[fmax_p, 1-fmax_p])[0]
    spec_config = {
        "preemphasis": None,
        "n_fft": n_fft, "hop_length": n_fft // 4, "win_length": n_fft, "window": "hann",
        "left_pad": pad, "right_pad": pad, "pad_mode": "reflect",  "center": center,
        "stft_power": 1,
        "mel_spec": True,
        "num_mels": 80, "fmin": fmin, "fmax": fmax
    }

    # generate post config
    log_base = random.choice([10, 'e'])
    log_factor = random.choice([20, 1])
    normalize_spec = random.choice([True, False])
    post_config = {
        "amp_to_db": True,
        "log_base": log_base,
        "log_factor": log_factor,
        "normalize_spec": normalize_spec,
        "ref_level_db": 0,
        "min_level_db": -100
    }

    config = {
        'github_repo': None, 'commit': None, 'wav_config': wav_config,
        'spec_config': spec_config, 'post_config': post_config}
    return config


def process(x):
    ext, n_iter, fpath, outdir = x
    y = ext.load(fpath)
    S = ext.convert(y)
    y_ = ext.inverse(S, n_iter=n_iter)
    idx = fpath.split('/')[-1].split('.')[0]
    outpath = os.path.join(outdir, f'{idx}.npy')
    np.save(outpath, y_, allow_pickle=False)


def generate(data, extension, n_cfg, n_iter, n_workers, outdir):
    # Search for files
    data_list = [os.path.join(data, x) for x in os.listdir(data) if x.endswith(extension)]
    print(f'{len(data_list)} {extension[1:]} files found in {data}')

    # Extract
    if len(data_list) != 0:
        os.makedirs(outdir, exist_ok=True)
        cfgdir = os.path.join(outdir, 'config')
        os.makedirs(cfgdir, exist_ok=True)
        n_workers = min(max(1, n_workers), cpu_count())
        for i in range(n_cfg):
            print(f'Processing config: {i+1}/{n_cfg}')
            # Build subdir
            subdir = os.path.join(outdir, str(i))
            os.makedirs(subdir, exist_ok=True)
            # Gen cfg and ext
            cfg = rand_config()
            ext = Extractor(cfg)
            with open(cfgdir+'/'+str(i)+'.json', 'w') as w:
                json.dump(cfg, w, indent=4)

            s = int((i/n_cfg)*len(data_list))
            e = int(((i+1)/n_cfg)*len(data_list))
            f_list = [(ext, n_iter, x, subdir) for x in data_list[s:e]]
            with Pool(processes=n_workers) as pool:
                for _ in pool.imap_unordered(process, tqdm(f_list)):
                    pass
        print('Completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess for TTS and vocoder')
    parser.add_argument('--data', '-d', metavar='DATA',
                        default='data', help='The dataset folder')
    parser.add_argument('--extension', '-e', metavar='EXT', default='.wav',
                        help='File extension to search for in dataset folder')
    parser.add_argument('--n_cfg', '-nc', metavar='N', type=int, default=4,
                        help='The number of random configs')
    parser.add_argument('--n_iter', '-ni', metavar='N', type=int, default=32,
                        help='The number of iterations for Griffin-Lim')
    parser.add_argument('--n_workers', '-n', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    parser.add_argument('--outdir', '-o', metavar='OUT', default='res',
                        help='Output directory')
    args = parser.parse_args()
    generate(**vars(args))