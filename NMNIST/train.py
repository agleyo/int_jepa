import argparse
from pathlib import Path

import torch
import torch.nn as nn


from NMNIST.helpers import (
    cosine_ema_schedule, ijepa_loss, 
    extract_latents, save_epoch_sample, train_probe, write_json, 
    write_metrics_csv, visualize_reconstruction, set_seed)

from NMNIST.data import prepare_dataloaders

from NMNIST.models import SpikingDecoder, IJEPA_SNN



def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    run_dir = Path(args.output_dir)
    sample_dir = run_dir / "epoch_latent_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    train_loader, train_eval_loader, test_loader, sample_loader = prepare_dataloaders(
        Path(args.data_root), args.batch_size, args.sample_size, args.num_workers, 
        args.seed, args.num_steps, args.bin_size, args.tau
    )

    model = IJEPA_SNN(latent_dim=args.latent_dim, num_steps=args.num_steps, beta=args.beta).to(device)
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    
    decoder = SpikingDecoder(latent_dim=args.latent_dim, num_steps=args.num_steps, beta=args.beta).to(device)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    total_steps = args.epochs * len(train_loader)
    global_step, best_probe_acc = 0, 0.0
    metrics = []

    config = vars(args).copy()
    config["device"] = str(device)
    write_json(run_dir / "config.json", config)

    for epoch in range(1, args.epochs + 1):
        model.train()
        decoder.train()
        epoch_total = epoch_align = epoch_var = epoch_dec = 0.0
        batches = 0

        for context_frames, target_frames, _ in train_loader:
            context_frames = context_frames.to(device, non_blocking=True)
            target_frames = target_frames.to(device, non_blocking=True)
            
            # Forward Pass
            z_context_seq = model.online_encoder(context_frames)
            pred_seq = model.predictor(z_context_seq)
            with torch.no_grad():
                z_target_seq = model.target_encoder(target_frames)
                
            z_context_pot = z_context_seq[-1]
            pred_pot = pred_seq[-1]
            z_target_pot = z_target_seq[-1]

            # JEPA Update
            loss, parts = ijepa_loss(pred_pot, z_target_pot, z_context_pot, args.var_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.online_encoder.parameters()) + list(model.predictor.parameters()), 1.0)
            optimizer.step()

            momentum = cosine_ema_schedule(args.ema_start, args.ema_end, global_step, total_steps)
            model.update_target_encoder(momentum)
            global_step += 1
            
            # Decoder Update (for visualization)
            reconstructed_frames = decoder(pred_seq.detach())
            target_frames_T = target_frames.transpose(0, 1).detach()
            dec_loss = mse_loss(reconstructed_frames, target_frames_T)
            
            decoder_optimizer.zero_grad(set_to_none=True)
            dec_loss.backward()
            decoder_optimizer.step()

            epoch_total += parts["total_loss"]
            epoch_align += parts["align_loss"]
            epoch_var += parts["var_loss"]
            epoch_dec += dec_loss.item()
            batches += 1

        save_epoch_sample(model.online_encoder, sample_loader, device, epoch, sample_dir)

        train_latents, train_labels = extract_latents(model.online_encoder, train_eval_loader, device)
        test_latents, test_labels = extract_latents(model.online_encoder, test_loader, device)
        probe_stats = train_probe(
            train_latents=train_latents, train_labels=train_labels,
            test_latents=test_latents, test_labels=test_labels,
            latent_dim=args.latent_dim, device=device, probe_epochs=args.probe_epochs,
            probe_batch_size=args.probe_batch_size, probe_lr=args.probe_lr, seed=args.seed + epoch,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": epoch_total / max(batches, 1),
            "train_align_loss": epoch_align / max(batches, 1),
            "train_var_loss": epoch_var / max(batches, 1),
            "decoder_loss": epoch_dec / max(batches, 1),
            "probe_final_acc": probe_stats["probe_final_acc"],
            "probe_best_acc": probe_stats["probe_best_acc"],
        }
        metrics.append(row)
        write_metrics_csv(run_dir / "metrics.csv", metrics)

        best_probe_acc = max(best_probe_acc, probe_stats["probe_best_acc"])
        torch.save(
            {"epoch": epoch, "online_encoder": model.online_encoder.state_dict(),
             "target_encoder": model.target_encoder.state_dict(), "predictor": model.predictor.state_dict(),
             "decoder": decoder.state_dict(),
             "optimizer": optimizer.state_dict(), "decoder_optimizer": decoder_optimizer.state_dict(),
             "metrics": metrics, "best_probe_acc": best_probe_acc},
            run_dir / "checkpoint_last.pt"
        )

        print(f"Epoch {epoch:03d} | JEPA={row['train_total_loss']:.4f} | Dec={row['decoder_loss']:.4f} | probe_acc={row['probe_final_acc'] * 100:.2f}%")
        
        # Visualization
        if epoch % 1 == 0 or epoch == args.epochs:
            visualize_reconstruction(model, decoder, test_loader, device, epoch, run_dir, num_samples=3)

    return run_dir

def parse_args():
    parser = argparse.ArgumentParser(description="SEW-ResNet Spiking Temporal I-JEPA on N-MNIST")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=16, help="Timesteps for SNN simulation")
    parser.add_argument("--bin-size", type=int, default=1024, help="Number of subsequent events in past and future bins")
    parser.add_argument("--tau", type=float, default=5000.0, help="Time constant for timesurfaces (in microseconds)")
    parser.add_argument("--beta", type=float, default=0.9, help="Membrane potential decay rate")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-start", type=float, default=0.99)
    parser.add_argument("--ema-end", type=float, default=0.9995)
    parser.add_argument("--var-weight", type=float, default=0.05)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)