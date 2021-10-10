# -*- coding: utf-8 -*-

import os
import torch
import math
import argparse
import torch.nn as nn
import numpy as np

from pathlib import Path
from multiprocessing import cpu_count
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from tensorboardX import SummaryWriter 
 

from dataset import AudioDataset
import adaptor
from utils.plot import plot_spec


def get_dataloader(data_dir, preprocess_dir, fix_config_dir, batch_size, n_workers, segment_length):
    """Generate dataloader"""
    dataset = AudioDataset(data_dir, preprocess_dir, fix_config_dir, segment_len=segment_length)

    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=None,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=None,
    )

    return train_loader, valid_loader


"""# Model Function
- Model forward function.
"""

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    inputs, gt, config = batch
    inputs = inputs.to(device)
    gt = gt.to(device)
    config = config.to(device)

    outs = model(inputs, config)
    if isinstance(outs, list):
        loss = 0.0
        for out in outs:
            loss = loss + criterion(out, gt)
        loss = loss / len(outs)
    else:
        ma = gt.max(1, True)[0].max(2, True)[0]
        mi = gt.min(1, True)[0].min(2, True)[0]
        loss = criterion(
                (outs-mi)/(ma-mi),
                (gt-mi)/(ma-mi))

    return loss, outs


"""# Validate
- Calculate accuracy of the validation set.
"""

def valid(dataloader, model, criterion, device): 
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, outs = model_fn(batch, model, criterion, device)
            running_loss += loss.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
        loss=f"{running_loss / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_loss / len(dataloader)


"""# Main function"""


def main(
    data_dir,
    preprocess_dir,
    fix_config_dir,
    out_dir,
    exp_name,
    batch_size,
    n_workers,
    segment_length,
    valid_steps,
):
    """Main function."""
    print(f"[Info]: Doing {exp_name} experiment!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader = get_dataloader(data_dir, preprocess_dir, fix_config_dir, batch_size, n_workers, segment_length)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    save_steps = valid_steps * 1
    total_steps = valid_steps * 100
    model = adaptor.Refiner_UNet_affine(n_channels=1, config_len=8, num_layers=4, base=16, bilinear=False, res_add=True).to(device)
    ckpt_file = os.path.join(out_dir, 'ckpts', f"{exp_name}.ckpt")
    if os.path.isfile(ckpt_file):
        model.load_state_dict(torch.load(ckpt_file))
        print("[Info]: Load model checkpoint!",flush = True)
    criterion = nn.L1Loss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
       optimizer, step_size = valid_steps * 50, gamma = 0.5)
    print(f"[Info]: Finish creating model!",flush = True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'log', exp_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        os.chmod(log_dir, 0o775)
    writer = SummaryWriter(log_dir)

    best_loss = 10000000
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # change data
        if (step + 1) % valid_steps == 1 and step != 0:
            train_loader, valid_loader = get_dataloader(data_dir, preprocess_dir, fix_config_dir, batch_size, n_workers, segment_length)
            train_iterator = iter(train_loader)
            print(f"[Info]: Finish Reloading data!",flush = True)
            
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, outs = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        writer.add_scalar('training_loss', loss, step)

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log
        pbar.update()
        pbar.set_postfix(
        loss=f"{batch_loss:.2f}",
        step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_loss = valid(valid_loader, model, criterion, device)
            writer.add_scalar('valid_loss', valid_loss, step)
            writer.add_image('input',
                plot_spec(batch[0][0].detach().cpu().numpy()), step)
            writer.add_image('target',
                plot_spec(batch[1][0].detach().cpu().numpy()), step)
            writer.add_image('output',
                plot_spec(outs[0].detach().cpu().numpy()), step)

            # keep the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")
            print(f"\n[Info]: Current lr:{optimizer.param_groups[0]['lr']}", flush = True)

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            ckpt_dir = os.path.join(out_dir, 'ckpts')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            save_path = os.path.join(ckpt_dir, f"{exp_name}.ckpt")
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (loss={best_loss:.4f})")

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='configs for training')  
    parser.add_argument('--data_dir', '-d', metavar='DATA',
                        default='data', help='The dataset folder')
    parser.add_argument('--preprocess_dir', '-p', metavar='DATA',
                        default='res', help='The preprocessed folder')
    parser.add_argument('--fix_config_dir', '-c', metavar='FILE', default='./config',
                        help='The config file for Extractor')
    parser.add_argument('--out_dir', '-o', metavar='OUT', default='./results',
                        help='Output directory')
    parser.add_argument('--exp_name', '-e', metavar='EXP', default='./exp',
                        help='Name of experiments')
    parser.add_argument('--n_workers', '-n', metavar='N', type=int,
                        default=cpu_count()-1,
                        help='The number of worker threads for preprocessing')
    parser.add_argument('--batch_size', '-b', metavar='BATCH', type=int,
                        default=32,  help='training batch size')
    parser.add_argument('--segment_length', '-s', metavar='SEG', type=int,
                        default=200,  help='training segment length')
    parser.add_argument('--valid_steps', '-v', metavar='VALID', type=int,
                        default=3600,  help='training segment length')
    args = parser.parse_args()
    main(**vars(args))
