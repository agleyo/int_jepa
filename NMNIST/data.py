from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import snntorch as snn
import tonic

class NMNISTTimeJEPA(torch.utils.data.Dataset):

    def __init__(self, save_to: str, train: bool = True, num_steps: int = 16, bin_size: int = 1024, tau: float = 5000.0):
        self.dataset = tonic.datasets.NMNIST(save_to=save_to, train=train)
        self.num_steps = num_steps
        self.bin_size = bin_size
        self.tau = tau
        self.sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
        
        # Cache valid indices so we don't have to scan 60k files every run
        cache_dir = Path(save_to)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"nmnist_valid_indices_{'train' if train else 'test'}_{bin_size}.pt"
        
        if cache_file.exists():
            print(f"Loading valid N-MNIST indices from {cache_file}...")
            self.valid_indices = torch.load(cache_file)
        else:
            self.valid_indices = []
            print(f"Filtering {'train' if train else 'test'} N-MNIST for >= {2*bin_size} events (this may take a minute)...")
            for i in range(len(self.dataset)):
                events, _ = self.dataset[i]
                if len(events) >= 2 * self.bin_size:
                    self.valid_indices.append(i)
            print(f"Kept {len(self.valid_indices)} out of {len(self.dataset)} samples.")
            torch.save(self.valid_indices, cache_file)

    def compute_timesurface(self, events: np.ndarray) -> np.ndarray:
        chunk_size = self.bin_size // self.num_steps
        # Array to store the last spike time for each pixel. Shape: (2, 34, 34) -> (P, W, H)
        last_times = np.full((self.sensor_size[2], self.sensor_size[0], self.sensor_size[1]), -np.inf, dtype=np.float32)
        
        frames = []
        for i in range(self.num_steps):
            # Extract the 16th sub-bin of events for the current timestep
            chunk = events[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) > 0:
                x = chunk['x'].astype(int)
                y = chunk['y'].astype(int)
                p = chunk['p'].astype(int)
                t = chunk['t'].astype(np.float32)
                
                # Update last seen spike times
                np.maximum.at(last_times, (p, x, y), t)
                # Reference time is the time of the latest event in this sub-bin
                t_ref = t[-1]
            else:
                # Fallback if chunk happens to be empty
                t_ref = np.max(last_times) if np.any(last_times != -np.inf) else 0.0
                
            # Timesurface decay: exp(-(t_ref - t_last) / tau)
            diff = (last_times - t_ref) / self.tau
            # Clip to prevent positive values (events in future) and handle -inf smoothly
            surface = np.exp(np.clip(diff, -np.inf, 0.0))
            frames.append(surface)
            
        return np.stack(frames)  # [num_steps, 2, 34, 34]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        events, target = self.dataset[real_idx]
        
        # Temporal slicing
        events_context = events[:self.bin_size]
        events_target = events[self.bin_size : 2 * self.bin_size]
        
        # Convert to sequences of timesurfaces: [T, 2, 34, 34]
        frames_context = self.compute_timesurface(events_context)
        frames_target = self.compute_timesurface(events_target)
        
        return frames_context, frames_target, target




def prepare_dataloaders(data_root: Path, batch_size: int, sample_size: int, num_workers: int, seed: int, num_steps: int, bin_size: int, tau: float):
    train_dataset = NMNISTTimeJEPA(save_to=str(data_root), train=True, num_steps=num_steps, bin_size=bin_size, tau=tau)
    test_dataset = NMNISTTimeJEPA(save_to=str(data_root), train=False, num_steps=num_steps, bin_size=bin_size, tau=tau)

    g = torch.Generator()
    g.manual_seed(seed)
    sample_indices = torch.randperm(len(test_dataset), generator=g)[:sample_size].tolist()
    sample_dataset = Subset(test_dataset, sample_indices)

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    sample_loader = DataLoader(sample_dataset, batch_size=sample_size, shuffle=False, **common)
    
    return train_loader, train_eval_loader, test_loader, sample_loader
