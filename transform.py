import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from extract import Extractor
from multiprocessing import Pool, cpu_count


def process(x):
    src_ext, tgt_ext, n_iter, fpath, outdir = x
    S = np.load(fpath)
    y = src_ext.inverse(S, n_iter=n_iter)
    S = tgt_ext.convert(y)
    idx = fpath.split('/')[-1].split('.')[0]
    outpath = os.path.join(outdir, f'{idx}.npy')
    np.save(outpath, S, allow_pickle=False)


def transform(src_config, tgt_config, data, extension,
             n_iter, n_workers, outdir):
    # Load config from json
    with open(src_config, 'r') as f:
        src_cfg = json.load(f)
    with open(tgt_config, 'r') as f:
        tgt_cfg = json.load(f)

    # Build Extractor
    src_ext = Extractor(src_cfg)
    tgt_ext = Extractor(tgt_cfg)

    # Search for files
    data_list = [os.path.join(data, x) for x in os.listdir(data) if x.endswith(extension)]
    f_list = [(src_ext, tgt_ext, n_iter, x, outdir) for x in data_list]
    print(f'{len(f_list)} {extension[1:]} files found in {data}')

    # Extract
    if len(f_list) != 0:
        os.makedirs(outdir, exist_ok=True)
        n_workers = min(max(1, n_workers), cpu_count())
        with Pool(processes=n_workers) as pool:
            for _ in pool.imap_unordered(process, tqdm(f_list)):
                pass
        print('Completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess for TTS and vocoder')
    parser.add_argument('--src_config', '-sc', metavar='FILE',
                        help='The config file of source features')
    parser.add_argument('--tgt_config', '-tc', metavar='FILE',
                        help='The config file of target features')
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
