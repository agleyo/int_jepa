import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
@torch.no_grad()
def cosine_ema_schedule(base_m: float, final_m: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return final_m
    progress = step / float(total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_m - (final_m - base_m) * cosine


def variance_regularizer(z: torch.Tensor, target_std: float = 0.2, eps: float = 1e-4) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def ijepa_loss(pred: torch.Tensor, target: torch.Tensor, z_context: torch.Tensor, var_weight: float) -> tuple[torch.Tensor, dict]:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target.detach(), dim=-1)
    align = 2.0 - 2.0 * (pred_n * target_n).sum(dim=-1).mean()
    var = variance_regularizer(z_context)
    total = align + var_weight * var
    return total, {"align_loss": align.item(), "var_loss": var.item(), "total_loss": total.item()}


@torch.no_grad()
def extract_latents(encoder: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_latents, all_labels = [], []
    for context_frames, _, labels in loader:
        context_frames = context_frames.to(device, non_blocking=True)
        latents = encoder(context_frames)[-1].cpu() 
        all_latents.append(latents)
        all_labels.append(labels.cpu())
    return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def save_epoch_sample(encoder: nn.Module, loader: DataLoader, device: torch.device, epoch: int, save_dir: Path) -> None:
    encoder.eval()
    context_frames, target_frames, labels = next(iter(loader))
    context_frames, target_frames = context_frames.to(device), target_frames.to(device)
    
    context_latents = encoder(context_frames)[-1].cpu().numpy()
    target_latents = encoder(target_frames)[-1].cpu().numpy()
    
    np.savez_compressed(
        save_dir / f"epoch_{epoch:03d}_latent_sample.npz",
        labels=labels.cpu().numpy(),
        target_latents=target_latents,
        context_latents=context_latents,
    )


@torch.no_grad()
def evaluate_probe(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

@torch.no_grad()
def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


@torch.no_grad()
def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

@torch.no_grad()
def normalize_img(img):
    img = img - img.min()
    max_val = img.max()
    return img / max_val if max_val > 0 else img

@torch.no_grad()
def visualize_reconstruction(model, decoder, dataloader, device, epoch, save_dir, num_samples=3):
    model.eval()
    decoder.eval()
    
    context_frames, target_frames, _ = next(iter(dataloader))
    context_frames = context_frames.to(device)

    z_context_seq = model.online_encoder(context_frames)
    pred_seq = model.predictor(z_context_seq)
    
    reconstructed_frames = decoder(pred_seq)
    target_frames_T = target_frames.transpose(0, 1).to(device)
    
    real_images = target_frames_T[-1].cpu().numpy()
    pred_images = reconstructed_frames[-1].cpu().numpy()

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    for i in range(num_samples):
        real_img = np.zeros((34, 34, 3))
        real_img[:, :, 0] = real_images[i, 0, :, :] # Red = Channel 0
        real_img[:, :, 2] = real_images[i, 1, :, :] # Blue = Channel 1
        
        pred_img = np.zeros((34, 34, 3))
        pred_img[:, :, 0] = pred_images[i, 0, :, :] 
        pred_img[:, :, 2] = pred_images[i, 1, :, :] 
        

        
        axes[i, 0].imshow(normalize_img(real_img))
        axes[i, 0].set_title(f"Real Timesurface")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(normalize_img(pred_img))
        axes[i, 1].set_title(f"Predicted Timesurface")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"nmnist_timesurface2_epoch_{epoch:03d}.png")
    plt.close()
