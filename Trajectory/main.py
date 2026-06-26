import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from simulation import simulate_world_streams
from models import CR_RPL_Passive_Model, Linear_JEPA_Model, NeuralODE_JEPA_Model, RAE_Model, init_weights_kaiming, variance_loss, compute_collapse_metrics, LinearPredictor, Predictor
from visualizations import run_parametric_inference, plot_parametric_window, plot_performance_and_robustness

def main():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_len = 256, 20
    canvas_size, glimpse_size = 64, 28
    latent_dim, rnn_hidden_dim = 16, 32
    epochs = 600
    train_noise_std = 0.1 

    models = {
        'jepa': CR_RPL_Passive_Model(latent_dim, rnn_hidden_dim).to(device),
        'lin_jepa': Linear_JEPA_Model(latent_dim, rnn_hidden_dim).to(device),
        'ode_jepa': NeuralODE_JEPA_Model(latent_dim, rnn_hidden_dim).to(device),
        'rae': RAE_Model(latent_dim, rnn_hidden_dim).to(device)
    }
    
    for m in models.values():
        m.apply(init_weights_kaiming)

    optimizers = {k: optim.AdamW(m.parameters(), lr=4e-4) for k, m in models.items()}
    shapes = ['bar', 'circle', 'triangle']

    print(f"Phase 1: Pre-training Representations (Noise Level: {train_noise_std})...")
    for epoch in range(1, epochs + 1):
        for opt in optimizers.values(): opt.zero_grad()
        
        shape_choice = shapes[epoch % len(shapes)]
        world_seq, _ = simulate_world_streams(batch_size, seq_len, canvas_size, device, 'normal', shape_choice, noise_std=train_noise_std)
        
        h_t = {k: torch.zeros(batch_size, rnn_hidden_dim, device=device) for k in models.keys()}
        losses = {k: 0.0 for k in models.keys()}
        metrics = {k: {'cos_sim': 0.0, 'collapse': 0.0} for k in ['jepa', 'lin_jepa', 'ode_jepa']}
        pos_weight = torch.tensor([20.0], device=device)

        for t in range(seq_len):
            g_P = F.interpolate(world_seq[:, t], size=(glimpse_size, glimpse_size), mode='area')
            
            for k in ['jepa', 'lin_jepa', 'ode_jepa']:
                z_P, z_P_hat_temp, h_t[k] = models[k](g_P, h_t[k])
                losses[k] += variance_loss(z_P) * 5.0
                if t > 0:
                    losses[k] += F.mse_loss(z_P_hat_temp, z_P)
                    metrics[k]['cos_sim'] += F.cosine_similarity(z_P_hat_temp, z_P, dim=-1).mean()
                metrics[k]['collapse'] += compute_collapse_metrics(z_P)

            x_hat, h_t['rae'] = models['rae'](g_P, h_t['rae'])
            losses['rae'] += F.binary_cross_entropy_with_logits(x_hat, g_P, pos_weight=pos_weight)

        for k in models.keys():
            (losses[k] / seq_len).backward()
            torch.nn.utils.clip_grad_norm_(models[k].parameters(), 1.0)
            optimizers[k].step()

        if epoch % 50 == 0 or epoch == 1:
            log_str = f"Ep {epoch:3d}/{epochs} | "
            for k in ['jepa', 'lin_jepa', 'ode_jepa']:
                log_str += f"{k.upper()} (Pred:{metrics[k]['cos_sim']/(seq_len-1):.2f}, Col:{metrics[k]['collapse']/seq_len:.2f}) | "
            log_str += f"RAE Loss: {losses['rae']/seq_len:.4f}"
            print(log_str)

    print("\nSaving models to 'outputs/models/'...")
    for k, m in models.items():
        torch.save(m.state_dict(), f"outputs/models/model_{k}.pth")

    print("\nPhase 2: Training Downstream Task on Noisy Data (Linear vs. Non-Linear Probes)...")
    for m in models.values(): m.eval()
    
    lin_probes = {k: LinearPredictor(rnn_hidden_dim, 3).to(device) for k in models.keys()}
    nonlin_probes = {k: Predictor(rnn_hidden_dim, 32, 3).to(device) for k in models.keys()}
    
    lin_probe_opts = {k: optim.Adam(p.parameters(), lr=1e-2) for k, p in lin_probes.items()}
    nonlin_probe_opts = {k: optim.Adam(p.parameters(), lr=1e-2) for k, p in nonlin_probes.items()}
    probe_epochs = 100

    final_lin_mse = {k: 0.0 for k in models.keys()}
    final_nonlin_mse = {k: 0.0 for k in models.keys()}

    for epoch in range(1, probe_epochs + 1):
        world_seq, gt = simulate_world_streams(batch_size, seq_len, canvas_size, device, 'normal', 'bar', noise_std=train_noise_std)
        h_t = {k: torch.zeros(batch_size, rnn_hidden_dim, device=device) for k in models.keys()}
        
        lin_probe_losses = {k: 0.0 for k in models.keys()}
        nonlin_probe_losses = {k: 0.0 for k in models.keys()}

        for opt in lin_probe_opts.values(): opt.zero_grad()
        for opt in nonlin_probe_opts.values(): opt.zero_grad()

        for t in range(seq_len):
            g_P = F.interpolate(world_seq[:, t], size=(glimpse_size, glimpse_size), mode='area')
            with torch.no_grad():
                for k in ['jepa', 'lin_jepa', 'ode_jepa']: _, _, h_t[k] = models[k](g_P, h_t[k])
                _, h_t['rae'] = models['rae'](g_P, h_t['rae'])
            
            for k in models.keys():
                pred_gt_lin = lin_probes[k](h_t[k].detach())
                lin_probe_losses[k] += F.mse_loss(pred_gt_lin, gt[:, t])
                
                pred_gt_nonlin = nonlin_probes[k](h_t[k].detach())
                nonlin_probe_losses[k] += F.mse_loss(pred_gt_nonlin, gt[:, t])

        for k in models.keys():
            (lin_probe_losses[k] / seq_len).backward()
            lin_probe_opts[k].step()
            
            (nonlin_probe_losses[k] / seq_len).backward()
            nonlin_probe_opts[k].step()
            
            if epoch == probe_epochs:
                final_lin_mse[k] = (lin_probe_losses[k] / seq_len).item()
                final_nonlin_mse[k] = (nonlin_probe_losses[k] / seq_len).item()
            
        if epoch == probe_epochs:
            print(f"{'Model':<12} | {'Linear MSE':<15} | {'Non-Linear MSE'}")
            print("-" * 45)
            for k in models.keys(): 
                print(f"{k.upper():<12} | {final_lin_mse[k]:<15.4f} | {final_nonlin_mse[k]:.4f}")

    print("\nEvaluating Robustness across Noise Levels...")
    noise_test_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    robustness_lin = {k: [] for k in models.keys()}
    robustness_nonlin = {k: [] for k in models.keys()}

    for nl in noise_test_levels:
        eval_lin_loss = {k: 0.0 for k in models.keys()}
        eval_nonlin_loss = {k: 0.0 for k in models.keys()}
        
        world_seq_eval, gt_eval = simulate_world_streams(batch_size, seq_len, canvas_size, device, 'normal', 'bar', noise_std=nl)
        h_t_eval = {k: torch.zeros(batch_size, rnn_hidden_dim, device=device) for k in models.keys()}
        
        with torch.no_grad():
            for t in range(seq_len):
                g_P = F.interpolate(world_seq_eval[:, t], size=(glimpse_size, glimpse_size), mode='area')
                
                for k in ['jepa', 'lin_jepa', 'ode_jepa']: 
                    _, _, h_t_eval[k] = models[k](g_P, h_t_eval[k])
                _, h_t_eval['rae'] = models['rae'](g_P, h_t_eval['rae'])
                
                for k in models.keys():
                    pred_lin = lin_probes[k](h_t_eval[k])
                    eval_lin_loss[k] += F.mse_loss(pred_lin, gt_eval[:, t]).item()
                    
                    pred_nonlin = nonlin_probes[k](h_t_eval[k])
                    eval_nonlin_loss[k] += F.mse_loss(pred_nonlin, gt_eval[:, t]).item()
                    
        for k in models.keys():
            robustness_lin[k].append(eval_lin_loss[k] / seq_len)
            robustness_nonlin[k].append(eval_nonlin_loss[k] / seq_len)

    plot_performance_and_robustness(
        final_lin_mse, final_nonlin_mse, 
        noise_test_levels, 
        robustness_lin, robustness_nonlin, 
        "outputs/visualizations/Window_3_Performance_Robustness.png"
    )

    print("\nPhase 3: Generating Visualizations (loading weights from disk)...")
    
    loaded_models = {
        'jepa': CR_RPL_Passive_Model(latent_dim, rnn_hidden_dim).to(device),
        'lin_jepa': Linear_JEPA_Model(latent_dim, rnn_hidden_dim).to(device),
        'ode_jepa': NeuralODE_JEPA_Model(latent_dim, rnn_hidden_dim).to(device),
        'rae': RAE_Model(latent_dim, rnn_hidden_dim).to(device)
    }
    for k, m in loaded_models.items():
        m.load_state_dict(torch.load(f"outputs/models/model_{k}.pth", weights_only=True))
        m.eval()
    
    scenarios = ['normal', 'sudden_stop', 'teleport', 'reverse', 'sine_wave', 'accelerate']
    all_h, all_h_pred = {k: {} for k in loaded_models.keys()}, {k: {} for k in loaded_models.keys()}
    all_gt_data = {}
    
    for scen in scenarios:
        w_seq, g_truth = simulate_world_streams(1, seq_len, canvas_size, device, movement_type=scen, shape_type='bar', noise_std=0.0)
        
        h_t = {k: torch.zeros(1, rnn_hidden_dim, device=device) for k in loaded_models.keys()}
        h_seq = {k: [] for k in loaded_models.keys()}
        h_seq_pred = {k: [] for k in loaded_models.keys()}
        
        with torch.no_grad():
            for t in range(seq_len):
                g_P = F.interpolate(w_seq[:, t], size=(glimpse_size, glimpse_size), mode='area')
                
                _, _, h_t['jepa'] = loaded_models['jepa'](g_P, h_t['jepa'])
                _, _, h_t['lin_jepa'] = loaded_models['lin_jepa'](g_P, h_t['lin_jepa'])
                _, _, h_t['ode_jepa'] = loaded_models['ode_jepa'](g_P, h_t['ode_jepa'])
                x_hat_logits_rae, h_t['rae'] = loaded_models['rae'](g_P, h_t['rae'])
                
                for k in loaded_models.keys():
                    h_seq[k].append(h_t[k].cpu().squeeze().numpy())
                
                z_hat = loaded_models['jepa'].temporal_predictor_P(h_t['jepa'])
                h_seq_pred['jepa'].append(loaded_models['jepa'].rnn(z_hat, h_t['jepa']).cpu().squeeze().numpy())

                z_hat_lin = loaded_models['lin_jepa'].temporal_predictor_P(h_t['lin_jepa'])
                h_seq_pred['lin_jepa'].append(loaded_models['lin_jepa'].rnn(z_hat_lin, h_t['lin_jepa']).cpu().squeeze().numpy())

                z_hat_ode = loaded_models['ode_jepa'].temporal_predictor_P(h_t['ode_jepa'])
                h_seq_pred['ode_jepa'].append(loaded_models['ode_jepa'].rnn(z_hat_ode, h_t['ode_jepa']).cpu().squeeze().numpy())
                
                x_hat_rae = torch.sigmoid(x_hat_logits_rae)
                z_hat_rae = loaded_models['rae'].enc(x_hat_rae)
                h_seq_pred['rae'].append(loaded_models['rae'].rnn(z_hat_rae, h_t['rae']).cpu().squeeze().numpy())
                
        for k in loaded_models.keys():
            all_h[k][scen] = np.array(h_seq[k])
            all_h_pred[k][scen] = np.array(h_seq_pred[k])
        all_gt_data[scen] = g_truth[0].cpu().numpy()

    pcas = {}
    all_projs = []
    for k in loaded_models.keys():
        stacked = np.vstack([np.vstack([all_h[k][s], all_h_pred[k][s]]) for s in scenarios])
        if stacked.var() < 1e-6: stacked += np.random.normal(0, 1e-4, stacked.shape)
        pcas[k] = PCA(n_components=2).fit(stacked)
        for scen in scenarios:
            all_projs.append(pcas[k].transform(all_h[k][scen]))
            all_projs.append(pcas[k].transform(all_h_pred[k][scen]))
            
    all_projs = np.vstack(all_projs)
    x_min, x_max = all_projs[:, 0].min(), all_projs[:, 0].max()
    y_min, y_max = all_projs[:, 1].min(), all_projs[:, 1].max()
    x_margin, y_margin = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05

    fig1, axes1 = plt.subplots(len(scenarios), 5, figsize=(30, 5 * len(scenarios)))
    fig1.tight_layout(pad=6.0)
    
    for idx, scen in enumerate(scenarios):
        gt = all_gt_data[scen]
        
        ax_phys = axes1[idx, 0]
        ax_phys.plot(range(seq_len), gt[:, 0], color='blue', label='X position', marker='o')
        ax_phys.plot(range(seq_len), np.sin(gt[:, 2]), color='orange', label='Sin(Angle)', linestyle='--')
        if idx == 0: ax_phys.legend(loc='upper right')
        ax_phys.set_title(f"Physics: [{scen.upper()}]", fontweight='bold')
        ax_phys.grid(True, alpha=0.3)
        ax_phys.axvline(x=seq_len//2, color='red', linestyle=':')

        for col, (k, title) in enumerate(zip(
            ['jepa', 'lin_jepa', 'ode_jepa', 'rae'],
            ['JEPA Latent', 'Linear JEPA', 'Neural ODE JEPA', 'RAE Latent']
        ), 1):
            ax = axes1[idx, col]
            proj = pcas[k].transform(all_h[k][scen])
            ax.plot(proj[:, 0], proj[:, 1], color='gray', linestyle='-', alpha=0.5, zorder=1)
            scatter = ax.scatter(proj[:, 0], proj[:, 1], c=range(seq_len), cmap='plasma', s=60, zorder=2)
            ax.scatter(proj[0, 0], proj[0, 1], color='green', marker='s', s=80, zorder=5)
            ax.scatter(proj[seq_len//2, 0], proj[seq_len//2, 1], color='red', marker='X', s=80, zorder=5)
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)

    cbar_ax1 = fig1.add_axes([1.01, 0.15, 0.02, 0.7])
    fig1.colorbar(scatter, cax=cbar_ax1, label='Temporal Time Step (t)')
    fig1.suptitle("Window 1: Global and Physics Dashboard", fontsize=22, y=1.02, fontweight='bold')
    plt.savefig("outputs/visualizations/Window_1_Dashboard.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2, axes2 = plt.subplots(4, len(scenarios), figsize=(6 * len(scenarios), 24))
    fig2.tight_layout(pad=5.0)
    arrow_style = dict(arrowstyle="-|>", color='magenta', lw=1.5, alpha=0.9, mutation_scale=12)
    
    for idx, scen in enumerate(scenarios):
        for row, (k, title) in enumerate(zip(
            ['jepa', 'lin_jepa', 'ode_jepa', 'rae'],
            ["JEPA Pred", "Linear JEPA Pred", "Neural ODE Pred", "RAE Pred"]
        )):
            proj = pcas[k].transform(all_h[k][scen])
            proj_pred = pcas[k].transform(all_h_pred[k][scen])
            
            ax = axes2[row, idx]
            ax.plot(proj[:, 0], proj[:, 1], color='gray', linestyle='-', alpha=0.3, zorder=1)
            scat2 = ax.scatter(proj[:, 0], proj[:, 1], c=range(seq_len), cmap='plasma', s=40, zorder=2)
            
            for t in range(seq_len):
                ax.annotate('', xy=(proj_pred[t, 0], proj_pred[t, 1]), 
                                   xytext=(proj[t, 0], proj[t, 1]), arrowprops=arrow_style)
            
            if idx == 0:
                ax.plot([], [], color='magenta', label='Prediction (t+1)', linewidth=1.5)
                ax.legend(loc='upper right')
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_title(f"{title}: [{scen.upper()}]", fontweight='bold')
            ax.grid(True, alpha=0.3)

    cbar_ax2 = fig2.add_axes([1.01, 0.15, 0.015, 0.7])
    fig2.colorbar(scat2, cax=cbar_ax2, label='Temporal Time Step (t)')
    fig2.suptitle("Window 2: Predicted Inference Trajectories (T to T+1)", fontsize=22, y=1.02, fontweight='bold')
    plt.savefig("outputs/visualizations/Window_2_Predictions.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    models_to_test = tuple(loaded_models.values())
    
    rot_configs = [
        {'name': 'Slow (w=0.1)', 'v': 0.06, 'w': 0.10, 'noise_std': 0.0},
        {'name': 'Normal (w=0.25)', 'v': 0.06, 'w': 0.25, 'noise_std': 0.0},
        {'name': 'Fast (w=0.45)', 'v': 0.06, 'w': 0.45, 'noise_std': 0.0}
    ]
    rot_results = run_parametric_inference(models_to_test, rot_configs, seq_len, canvas_size, glimpse_size, rnn_hidden_dim, device)
    plot_parametric_window(rot_results, "Window 4: Sensitivity to Rotations (Fixed v=0.06)", seq_len, "outputs/visualizations/Window_4_Rotations.png")

    mov_configs = [
        {'name': 'Slow (v=0.03)', 'v': 0.03, 'w': 0.25, 'noise_std': 0.0},
        {'name': 'Normal (v=0.06)', 'v': 0.06, 'w': 0.25, 'noise_std': 0.0},
        {'name': 'Fast (v=0.10)', 'v': 0.10, 'w': 0.25, 'noise_std': 0.0}
    ]
    mov_results = run_parametric_inference(models_to_test, mov_configs, seq_len, canvas_size, glimpse_size, rnn_hidden_dim, device)
    plot_parametric_window(mov_results, "Window 5: Sensitivity to Movements (Fixed w=0.25)", seq_len, "outputs/visualizations/Window_5_Movements.png")

    noise_configs = [
        {'name': 'Clean (noise=0.0)', 'v': 0.06, 'w': 0.25, 'noise_std': 0.0},
        {'name': 'Mild Noise (noise=0.2)', 'v': 0.06, 'w': 0.25, 'noise_std': 0.2},
        {'name': 'Heavy Noise (noise=0.5)', 'v': 0.06, 'w': 0.25, 'noise_std': 0.5}
    ]
    noise_results = run_parametric_inference(models_to_test, noise_configs, seq_len, canvas_size, glimpse_size, rnn_hidden_dim, device)
    plot_parametric_window(noise_results, "Window 6: Sensitivity to Noise Injection", seq_len, "outputs/visualizations/Window_6_Noise.png")

    print("\nProcess Complete! Visualizations are available in 'outputs/visualizations/'.")

if __name__ == "__main__":
    main()