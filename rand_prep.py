import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from extract import Extractor
from dataset import generate_config
from multiprocessing import Pool, cpu_count


def process(x):
    ext, n_iter, fpath, outdir = x
    y = ext.load(fpath)
    S = ext.convert(y)
    S = ext.post_convert(S)
    y_ = ext.inverse(S, n_iter=n_iter)
    idx = fpath.split('/')[-1].split('.')[0]
    outpath = os.path.join(outdir, f'{idx}.npy')
    np.save(outpath, y_, allow_pickle=False)


def generate(data, extension, n_cfg, cfg_dir, n_iter, n_workers, outdir):
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
            cfg = generate_config(cfg_dir)
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
    parser.add_argument('--cfg_dir', '-cfg', metavar='N', type=str, default='./config',
                        help='dir of fixed configs')
    parser.add_argument('--n_iter', '-ni', metavar='N', type=int, default=32,
                        help='The number of iterations for Griffin-Lim')
    parser.add_argument('--n_workers', '-n', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    parser.add_argument('--outdir', '-o', metavar='OUT', default='res',
                        help='Output directory')
    args = parser.parse_args()
    generate(**vars(args))