import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from simulation import simulate_world_streams

def run_parametric_inference(models, config_list, seq_len, canvas_size, glimpse_size, rnn_hidden_dim, device):
    model_jepa, model_lin_jepa, model_ode_jepa, model_rae = models
    results = {}
    for conf in config_list:
        name = conf['name']
        noise_val = conf.get('noise_std', 0.0)
        w_seq, g_truth = simulate_world_streams(
            1, seq_len, canvas_size, device, 'normal', 'bar', conf['v'], conf['w'], noise_std=noise_val
        )
        
        h_t_jepa = torch.zeros(1, rnn_hidden_dim, device=device)
        h_t_lin_jepa = torch.zeros(1, rnn_hidden_dim, device=device)
        h_t_ode_jepa = torch.zeros(1, rnn_hidden_dim, device=device)
        h_t_rae = torch.zeros(1, rnn_hidden_dim, device=device)
        
        h_seq_jepa, h_seq_lin_jepa, h_seq_ode_jepa, h_seq_rae = [], [], [], []
        
        with torch.no_grad():
            for t in range(seq_len):
                g_P = F.interpolate(w_seq[:, t], size=(glimpse_size, glimpse_size), mode='area')
                _, _, h_t_jepa = model_jepa(g_P, h_t_jepa)
                _, _, h_t_lin_jepa = model_lin_jepa(g_P, h_t_lin_jepa)
                _, _, h_t_ode_jepa = model_ode_jepa(g_P, h_t_ode_jepa)
                _, h_t_rae = model_rae(g_P, h_t_rae)
                
                h_seq_jepa.append(h_t_jepa.cpu().squeeze().numpy())
                h_seq_lin_jepa.append(h_t_lin_jepa.cpu().squeeze().numpy())
                h_seq_ode_jepa.append(h_t_ode_jepa.cpu().squeeze().numpy())
                h_seq_rae.append(h_t_rae.cpu().squeeze().numpy())
                
        results[name] = {
            'jepa': np.array(h_seq_jepa), 'lin_jepa': np.array(h_seq_lin_jepa),
            'ode_jepa': np.array(h_seq_ode_jepa), 'rae': np.array(h_seq_rae),
            'gt': g_truth[0].cpu().numpy()
        }
    return results

def plot_parametric_window(results, title_text, seq_len, out_path):
    names = list(results.keys())
    fig, axes = plt.subplots(5, len(names), figsize=(5 * len(names), 20))
    fig.tight_layout(pad=6.0)
    
    pcas = {}
    all_projs = []
    for k in ['jepa', 'lin_jepa', 'ode_jepa', 'rae']:
        stacked = np.vstack([results[n][k] for n in names])
        if stacked.var() < 1e-6: stacked += np.random.normal(0, 1e-4, stacked.shape)
        pcas[k] = PCA(n_components=2).fit(stacked)
        for n in names:
            all_projs.append(pcas[k].transform(results[n][k]))
            
    all_projs = np.vstack(all_projs)
    x_min, x_max = all_projs[:, 0].min(), all_projs[:, 0].max()
    y_min, y_max = all_projs[:, 1].min(), all_projs[:, 1].max()
    x_margin, y_margin = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    
    for idx, name in enumerate(names):
        gt = results[name]['gt']
        
        ax_phys = axes[0, idx]
        ax_phys.plot(range(seq_len), gt[:, 0], color='blue', label='Position X', marker='o')
        ax_phys.plot(range(seq_len), np.sin(gt[:, 2]), color='orange', label='Sin(Angle)', linestyle='--')
        if idx == 0: ax_phys.legend()
        ax_phys.set_title(f"Physics: {name}", fontweight='bold')
        ax_phys.grid(True, alpha=0.3)

        for row, (k, title) in enumerate(zip(
            ['jepa', 'lin_jepa', 'ode_jepa', 'rae'],
            ['JEPA Latent', 'Linear JEPA', 'Neural ODE JEPA', 'RAE Latent']
        ), 1):
            ax = axes[row, idx]
            proj = pcas[k].transform(results[name][k])
            ax.plot(proj[:, 0], proj[:, 1], color='gray', linestyle='-', alpha=0.5, zorder=1)
            ax.scatter(proj[:, 0], proj[:, 1], c=range(seq_len), cmap='plasma', s=60, zorder=2)
            ax.scatter(proj[0, 0], proj[0, 1], color='green', marker='s', s=80, zorder=5)
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    fig.suptitle(title_text, fontsize=20, y=0.98, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_performance_and_robustness(lin_mse, nonlin_mse, noise_levels, robust_lin, robust_nonlin, out_path):
    models = list(lin_mse.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.tight_layout(pad=6.0)

    ax = axes[0]
    lin_vals = [lin_mse[k] for k in models]
    nonlin_vals = [nonlin_mse[k] for k in models]
    
    ax.bar(x - width/2, lin_vals, width, label='Linear Probe', color='skyblue')
    ax.bar(x + width/2, nonlin_vals, width, label='Non-Linear Probe', color='salmon')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Downstream Task Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in models])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    for k in models:
        ax.plot(noise_levels, robust_lin[k], marker='o', label=k.upper())
    ax.set_xlabel('Noise Standard Deviation')
    ax.set_ylabel('Linear Probe MSE')
    ax.set_title('Linear Robustness vs. Noise', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for k in models:
        ax.plot(noise_levels, robust_nonlin[k], marker='s', label=k.upper(), linestyle='--')
    ax.set_xlabel('Noise Standard Deviation')
    ax.set_ylabel('Non-Linear Probe MSE')
    ax.set_title('Non-Linear Robustness vs. Noise', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Window 3: Probing Performance and Noise Robustness", fontsize=18, y=1.05, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()