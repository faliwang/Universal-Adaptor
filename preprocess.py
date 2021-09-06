import os
import argparse
from itertools import permutations, product
from multiprocessing import Pool, cpu_count

from generate import generate
from transform import transform


def preprocess(config_dir, data, n_iter, n_workers, outdir):
    config_dir = os.path.abspath(config_dir)
    configs = get_configs(config_dir)
    config_names = [config_name for _, config_name in configs]

    # generate ground truth
    gt_dir = os.path.join(outdir, 'ground_truth')
    os.makedirs(gt_dir, exist_ok=True)
    for config_path, config_name in configs:
        out_dir = os.path.join(gt_dir, config_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing npy in {out_dir}\n")
        generate(config_path, data, '.wav', n_workers, out_dir)

    # generate inputs
    input_dir = os.path.join(outdir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    perms = list(product(config_names, repeat=2))
    for perm in perms:
        src, tgt = perm
        data_dir = os.path.join(gt_dir, src)
        for config, config_name in configs:
            if config_name == tgt:
                tgt_config = config
            if config_name == src:
                src_config = config
        out_dir = os.path.join(input_dir, src + '_' + tgt)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing npy in {out_dir}\n")
        transform(src_config, tgt_config, data_dir, '.npy', n_iter, n_workers, out_dir)
    
    print('\nCompleted All!\n')
    

def get_configs(config_dir):
    """
        return list: [(config_path, config_name)]
    """
    config_dir = os.path.abspath(config_dir)
    config_paths = os.listdir(config_dir)
    config_paths.sort()
    configs = []
    for config_path in config_paths:
        config_name = config_path[:-5].split('_')[-1]
        config_path = os.path.join(config_dir, config_path)
        configs.append((config_path, config_name))
    return configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess for TTS and vocoder')
    parser.add_argument('--config_dir', '-c', metavar='DATA',
                        help='The config file for Extractor')
    parser.add_argument('--data', '-d', metavar='DATA',
                        default='data', help='The dataset folder')
    parser.add_argument('--n_iter', '-ni', metavar='N', type=int, default=32,
                        help='The number of iterations for Griffin-Lim')
    parser.add_argument('--n_workers', '-n', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    parser.add_argument('--outdir', '-o', metavar='OUT', default='res',
                        help='Output directory')
    args = parser.parse_args()
    preprocess(**vars(args))
