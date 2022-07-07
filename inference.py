# -*- coding: utf-8 -*-

import os
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from multiprocessing import cpu_count

from dataset import InferenceDataset, get_configs
from extract import Extractor
import adaptor


def main(
    data_dir,
    config_dir,
    exp_name,
    out_dir,
    num_workers,
):
    """Main function."""
    print(f"[Info]: Doing {exp_name} inference!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    data_dir = os.path.abspath(data_dir)
    config_dir = os.path.abspath(config_dir)
    configs = get_configs(config_dir)
    extractors = [(Extractor(config), config_name) for config, config_name in configs]
    dataset = InferenceDataset(data_dir, config_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=None,
    )
    print(f"[Info]: Finish loading data!",flush = True)

    model = adaptor.Refiner_UNet_affine(n_channels=1, config_len=8, num_layers=4, base=16, bilinear=False, res_add=True).to(device)
    if 'ckpts' not in out_dir:
        model_path = os.path.join(os.getcwd(), 'models', 'adaptor.ckpt')
    else:
        model_path = os.path.join(out_dir, 'ckpts', f'{exp_name}.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!",flush = True)

    output_dir = os.path.join(out_dir, 'mels_save', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dirs = {}
    models = os.listdir(data_dir)
    for model_name in models:
        model_dirs[model_name] = os.path.join(output_dir, model_name)
        if not os.path.exists(model_dirs[model_name]):
            os.makedirs(model_dirs[model_name])
    for mels, cfg_tensors, model_names, input_names in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            cfg_tensors = cfg_tensors.to(device)
            preds = model(mels, cfg_tensors)
            if isinstance(preds, list):
                preds = preds[-1].cpu().numpy()
            else:
                preds = preds.cpu().numpy()
            for pred, model_name, input_name in zip(preds, model_names, input_names):
                input_name = input_name[:-4]
                tgt_ext = None
                for ext, config_name in extractors:
                    if config_name == model_name.split('_')[-1]:
                        tgt_ext = ext
                        break
                pred = tgt_ext.post_convert(pred)
                np.save(os.path.join(model_dirs[model_name], input_name), pred)
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='configs for training')  
    parser.add_argument('--data_dir', '-d', metavar='DATA',
                        default='data', help='The dataset folder')
    parser.add_argument('--config_dir', '-c', metavar='FILE', default='./config',
                        help='The config file for Extractor')
    parser.add_argument('--out_dir', '-o', metavar='OUT', default='./results',
                        help='Output directory')
    parser.add_argument('--exp_name', '-e', metavar='EXP', default='./exp',
                        help='Name of experiments')
    parser.add_argument('--num_workers', '-n', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    args = parser.parse_args()
    main(**vars(args))