'''import argparse
import copy
import csv
import json
from json import encoder
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import snntorch as snn
from snntorch import surrogate
import tonic'''

import copy
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

class SEWBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, beta: float = 0.9, spike_grad=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor, mem1: torch.Tensor, mem2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(out, mem1)

        out = self.bn2(self.conv2(spk1))
        spk2, mem2 = self.lif2(out, mem2)

        out_spk = spk2 + self.shortcut(x)
        return out_spk, mem1, mem2

class SpikingDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, num_steps: int = 16, beta: float = 0.9):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc = nn.Linear(latent_dim, 128 * 5 * 5)
        self.lif_fc = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.sew1 = SEWBlock(64, 64, stride=1, beta=beta, spike_grad=spike_grad)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.sew2 = SEWBlock(32, 32, stride=1, beta=beta, spike_grad=spike_grad)
        
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.sew3 = SEWBlock(16, 16, stride=1, beta=beta, spike_grad=spike_grad)
        
        # Non-spiking layer
        self.conv_out = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.lif_out = snn.Leaky(beta=beta, reset_mechanism="none")
        
    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        T, B, _ = z_seq.shape
        
        mem_fc = self.lif_fc.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem_sew1_1, mem_sew1_2 = self.sew1.lif1.init_leaky(), self.sew1.lif2.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_sew2_1, mem_sew2_2 = self.sew2.lif1.init_leaky(), self.sew2.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_sew3_1, mem_sew3_2 = self.sew3.lif1.init_leaky(), self.sew3.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        out_frames = []
        for t in range(T):
            x = self.fc(z_seq[t])
            spk_fc, mem_fc = self.lif_fc(x, mem_fc)
            x_spatial = spk_fc.view(B, 128, 5, 5)
            
            up1_out = self.bn1(self.up1(x_spatial))
            spk1, mem1 = self.lif1(up1_out, mem1)
            spk_sew1, mem_sew1_1, mem_sew1_2 = self.sew1(spk1, mem_sew1_1, mem_sew1_2)
            
            up2_out = self.bn2(self.up2(spk_sew1))
            spk2, mem2 = self.lif2(up2_out, mem2)
            spk_sew2, mem_sew2_1, mem_sew2_2 = self.sew2(spk2, mem_sew2_1, mem_sew2_2)
            
            up3_out = self.bn3(self.up3(spk_sew2))
            spk3, mem3 = self.lif3(up3_out, mem3)
            spk_sew3, mem_sew3_1, mem_sew3_2 = self.sew3(spk3, mem_sew3_1, mem_sew3_2)
            
            # Non-spiking layer
            out_final = self.conv_out(spk_sew3)
            _, mem_out = self.lif_out(out_final, mem_out)
            out_frames.append(mem_out)
            
        return torch.stack(out_frames, dim=0)

class SEWResNetEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, num_steps: int = 16, beta: float = 0.9):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        spike_grad = surrogate.fast_sigmoid(slope=25)

        n_channels = 16
        self.conv1 = nn.Conv2d(2, n_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.layer1 = SEWBlock(n_channels, 32, stride=2, beta=beta, spike_grad=spike_grad)
        self.layer2 = SEWBlock(32, 64, stride=2, beta=beta, spike_grad=spike_grad)
        self.layer3 = SEWBlock(64, 128, stride=2, beta=beta, spike_grad=spike_grad)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, latent_dim)
        
        # Non-spiking readout
        self.lif_out = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        B = x.size(1)

        mem1 = self.lif1.init_leaky()
        mem_l1_1, mem_l1_2 = self.layer1.lif1.init_leaky(), self.layer1.lif2.init_leaky()
        mem_l2_1, mem_l2_2 = self.layer2.lif1.init_leaky(), self.layer2.lif2.init_leaky()
        mem_l3_1, mem_l3_2 = self.layer3.lif1.init_leaky(), self.layer3.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        mem_out_seq = []

        for step in range(self.num_steps):
            x_t = x[step]

            out = self.bn1(self.conv1(x_t))
            spk1, mem1 = self.lif1(out, mem1)

            spk2, mem_l1_1, mem_l1_2 = self.layer1(spk1, mem_l1_1, mem_l1_2)
            spk3, mem_l2_1, mem_l2_2 = self.layer2(spk2, mem_l2_1, mem_l2_2)
            spk4, mem_l3_1, mem_l3_2 = self.layer3(spk3, mem_l3_1, mem_l3_2)

            out = self.pool(spk4).view(B, -1)
            out = self.fc(out)
            
            # INon spiking latents
            _, mem_out = self.lif_out(out, mem_out)
            mem_out_seq.append(mem_out)
            
        return torch.stack(mem_out_seq, dim=0)

class SpikingPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, beta: float = 0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T, B, _ = x_seq.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        out_seq = []
        for t in range(T):
            out = self.fc1(x_seq[t])
            spk1, mem1 = self.lif1(out, mem1)
            
            out = self.fc2(spk1)
            _, mem2 = self.lif2(out, mem2) 
            
            out_seq.append(mem2)  

        return torch.stack(out_seq, dim=0)

class LatentProbe(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class IJEPA_SNN(nn.Module):
    def __init__(self, latent_dim: int = 128, num_steps: int = 16, beta: float = 0.9):
        super().__init__()
        self.online_encoder = SEWResNetEncoder(latent_dim=latent_dim, num_steps=num_steps, beta=beta)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = SpikingPredictor(latent_dim=latent_dim, hidden_dim=latent_dim * 2, beta=beta)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)

    def forward(self, context_frames: torch.Tensor, target_frames: torch.Tensor):
        z_context_seq = self.online_encoder(context_frames)
        pred_seq = self.predictor(z_context_seq)
        with torch.no_grad():
            z_target_seq = self.target_encoder(target_frames)
            
        return z_context_seq[-1], pred_seq[-1], z_target_seq[-1] 
