import torch
import torch.nn as nn
import numpy as np

from .unet import UNet
from .transformer.Layers import FFTBlock

class Refiner_UNet(nn.Module):
    def __init__(self, n_channels=1, num_layers=4, base=16, bilinear=True, res_add=False):
        super().__init__()
        self.unet = UNet(n_channels=n_channels, num_layers=num_layers, base=base, bilinear=bilinear, res_add=res_add)
    
    def forward(self, input, config):
        input_w = torch.unsqueeze(input, -1)
        input_w = torch.transpose(input_w, 1, 3)
        # B * chan * n_mels * len
        output = self.unet(input_w)
        output = torch.transpose(output, 1, 3)
        output = torch.squeeze(output, -1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output


class Refiner_UNet_with_config(nn.Module):
    def __init__(self, n_channels=1, num_layers=4, base=16, bilinear=True, res_add=False):
        super().__init__()
        self.unet = UNet(n_channels=n_channels, num_layers=num_layers, base=base, bilinear=bilinear, res_add=res_add)
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1)
    
    def forward(self, input, config):
        batch, num_mels, length = input.shape
        input_w = torch.unsqueeze(input, 1)
        # B * chan * n_mels * len
        config = torch.unsqueeze(config, -1)
        config = torch.unsqueeze(config, -1)
        config = config.repeat(1, 1, num_mels, length)
        input_u = torch.cat([input_w, config], 1)
        output = self.unet(input_u)
        output = self.conv(output) + input_w
        output = torch.squeeze(output, 1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output


def get_mask_from_lengths(lengths, device, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Refiner_FFT(nn.Module):
    def __init__(self, 
            device,
            max_seq_length = 1000,
            d_model=80,
            n_head=2,
            conv_filter_size=1024,
            conv_kernel_size= [9, 1],
            dropout=0.2,
            n_layers = 4,
        ):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        n_position = max_seq_length + 1
        d_word_vec = d_model
        n_head = n_head
        d_k = d_v = d_model // n_head
        d_inner = conv_filter_size
        kernel_size =  conv_kernel_size
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, mels, config, return_attns=False):
        """
        args:
        mels: (batch size, length, num_mel)
        return:
        out: (batch size, n_spks)
        """
        lens = mels.shape[-1]
        mask = get_mask_from_lengths(lens, self.device)

        dec_slf_attn_list = []
        batch_size, max_len = mels.shape[0], mels.shape[1]

        # -- Forward
        if not self.training and mels.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = mels + get_sinusoid_encoding_table(
            mels.shape[1], self.d_model
            )[: mels.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                mels.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = mels[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
        return dec_output, mask