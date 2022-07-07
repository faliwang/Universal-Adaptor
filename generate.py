import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from extract import Extractor
from multiprocessing import Pool, cpu_count


def process(x):
    ext, fpath, outdir = x
    y = ext.load(fpath)
    S = ext.convert(y)
    S = ext.post_convert(S)
    idx = fpath.split('/')[-1].split('.')[0]
    outpath = os.path.join(outdir, f'{idx}.npy')
    np.save(outpath, S, allow_pickle=False)


def generate(config, data, extension, n_workers, outdir):
    # Load config from json
    cfgs = []
    if config == 'all':
        configs = os.listdir('./config')
        configs.sort()
        for config in configs:
            config_json = os.path.join(os.getcwd(), 'config', config)
            with open(config_json, 'r') as f:
                cfg = json.load(f)
                cfgs.append(cfg)
    else:
        with open(config, 'r') as f:
            cfg = json.load(f)
            cfgs.append(cfg)

    # Build Extractor
    exts = []
    for cfg in cfgs:
        ext = Extractor(cfg)
        exts.append(ext)

    # Search for files
    data_list = [os.path.join(data, x) for x in os.listdir(data) if x.endswith(extension)]
    print(f'{len(data_list)} {extension[1:]} files found in {data}')
    f_list = []
    for i in range(len(configs)):
        config_name = configs[i][:-5].split('_')[1]
        out_dir = os.path.join(outdir, config_name)
        os.makedirs(out_dir, exist_ok=True)
        _list = [(exts[i], x, out_dir) for x in data_list]
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
    parser.add_argument('--config', '-c', metavar='FILE',
                        default='all', help='The config file for Extractor')
    parser.add_argument('--data', '-d', metavar='DATA',
                        default='data', help='The dataset folder')
    parser.add_argument('--extension', '-e', metavar='EXT', default='.wav',
                        help='File extension to search for in dataset folder')
    parser.add_argument('--n_workers', '-n', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    parser.add_argument('--outdir', '-o', metavar='OUT', default='res',
                        help='Output directory')
    args = parser.parse_args()
    generate(**vars(args))
