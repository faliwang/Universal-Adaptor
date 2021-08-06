# -*- coding: utf-8 -*-

import os
import torch
import math
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
 

from dataset import AudioDataset
from adaptor.models import Refiner_UNet, Refiner_FFT, Refiner_UNet_with_config


def get_dataloader(data_dir, data_type, config_dir, batch_size, n_workers, segment_length):
    """Generate dataloader"""
    dataset = AudioDataset(data_dir, data_type, config_dir, segment_len=segment_length)

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



"""# Learning rate schedule
- For transformer architecture, the design of learning rate schedule is different from that of CNN.
- Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
- The warmup schedule
  - Set learning rate to 0 in the beginning.
  - The learning rate increases linearly from 0 to initial learning rate during warmup period.
"""

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
        )
        return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

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

    loss = criterion(outs, gt)

    return loss


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
            loss = model_fn(batch, model, criterion, device)
            running_loss += loss.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
        loss=f"{running_loss / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_loss / len(dataloader)


"""# Main function"""

def parse_args():
    """arguments"""
    config = {
        "data_dir": "/work/b07502172/universal_adaptor/mels",
        "data_type": "npy",
        "config_dir": "./config",
        "out_dir": "/home/b07502172/universal_adaptor/Acoustic-feature-converter/results",
        "batch_size": 32,
        "n_workers": 4,
        "segment_length": 200,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 150000,
    }

    return config


def main(
    data_dir,
    data_type,
    config_dir,
    out_dir,
    batch_size,
    n_workers,
    segment_length,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader = get_dataloader(data_dir, data_type, config_dir, batch_size, n_workers, segment_length)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    model = Refiner_UNet_with_config(n_channels=20, num_layers=3, base=16, bilinear=False, res_add=True).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!",flush = True)

    best_loss = 10000000
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()

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

            # keep the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_path = os.path.join(out_dir, "model.ckpt")
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (loss={best_loss:.4f})")

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())