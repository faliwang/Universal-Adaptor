# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from dataset import InferenceDataset
from adaptor.models import Refiner_UNet, Refiner_FFT, Refiner_UNet_with_config


def parse_args():
    """arguments"""
    config = {
        "data_dir": "/work/b07502172/universal_adaptor/trial/mels/inference_trial",
        "config_dir": "./config",
        "model_path": "/home/b07502172/universal_adaptor/Acoustic-feature-converter/results/model_unet_3layer_base32_resadd_config.ckpt",
        "output_dir": "/home/b07502172/universal_adaptor/Acoustic-feature-converter/results/mels_save",
        "num_workers": 4,
    }

    return config


def main(
    data_dir,
    config_dir,
    model_path,
    output_dir,
    num_workers,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    data_dir = os.path.abspath(data_dir)
    config_dir = os.path.abspath(config_dir)
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

    model = Refiner_UNet_with_config(n_channels=20, num_layers=3, base=32, bilinear=False, res_add=True).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!",flush = True)

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
            preds = model(mels, cfg_tensors).cpu().numpy()
            for pred, model_name, input_name in zip(preds, model_names, input_names):
                input_name = input_name[:-4]
                if model_name.split('_')[-1] == 'wavernn':
                    pred = np.clip(pred, 0, 1)
                np.save(os.path.join(model_dirs[model_name], input_name), pred)
  


if __name__ == "__main__":
  main(**parse_args())