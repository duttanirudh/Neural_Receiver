"""
============================================================================
ViterbiNet + DeepRx: Hybrid Two-Stage Training Pipeline
============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from model import ViterbiNetDetector, DeepRxLoss, compute_ber, Phase, MEMORY_LENGTH, DEVICE
from data_generator import DeepRxDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to: {seed}")


class WarmupCosineScheduler:
    def __init__(self, optimizer, max_lr=1e-3, min_lr=1e-6, total_steps=10000, warmup_steps=500):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * (step / max(self.warmup_steps, 1))
        else:
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_step += 1
        return lr


class ViterbiNetTrainer:
    DEFAULT_CONFIG = {
        'max_lr': 1e-3,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'warmup_steps': 500,
        'batch_size': 16,
        'total_steps': 40000,
        'phase2_step': 25000,
        'val_every': 500,
        'log_every': 50,
        'grad_clip': 1.0,
        'num_workers': 0,
        'modulation': '16QAM',
        'save_dir': 'checkpoints',
        'experiment_name': 'viterbinet_deeprx',
        'seed': 42,
    }

    def __init__(self, model, train_dataset, val_dataset, config=None, device=DEVICE):
        self.device = device
        self.model = model.to(device)

        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        set_seed(self.config['seed'])

        self.criterion = DeepRxLoss().to(device)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'],
            shuffle=True, num_workers=self.config['num_workers'], drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'],
            shuffle=False, num_workers=self.config['num_workers'], drop_last=False
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config['max_lr'],
            weight_decay=self.config['weight_decay'], betas=(0.9, 0.999)
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer, max_lr=self.config['max_lr'], min_lr=self.config['min_lr'],
            total_steps=self.config['total_steps'], warmup_steps=self.config['warmup_steps']
        )

        self.history = {
            'train_loss': [], 'train_ber': [],
            'val_loss': [], 'val_ber': [],
            'learning_rates': [], 'steps': [], 'val_steps': []
        }

        self.best_val_ber = float('inf')
        self.save_dir = os.path.join(self.config['save_dir'], self.config['experiment_name'])
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        cfg = self.config
        print("\n" + "=" * 70)
        print(f"{'ViterbiNet 2-Stage Training — Run Started':^70}")
        print("=" * 70)
        print(f"  Device:       {self.device}")
        print(f"  Batch size:   {cfg['batch_size']}")
        print(f"  Total steps:  {cfg['total_steps']} (Phase 2 at {cfg['phase2_step']})")
        print("=" * 70)

        step = 0
        running_loss = 0.0
        running_ber = 0.0
        running_count = 0
        train_start = time.time()

        while step < cfg['total_steps']:
            for batch_data in self.train_loader:
                if step >= cfg['total_steps']:
                    break

                if step == cfg['phase2_step']:
                    print("\n" + "!"*70)
                    print(f"  PHASE 2 INITIATED: Shifting to High-SNR Fine-Tuning (10dB - 30dB)")
                    print("!"*70 + "\n")

                    phase2_dataset = DeepRxDataset(
                        n_samples=10000, n_rx_antennas=self.train_loader.dataset.n_rx_antennas,
                        modulation=cfg['modulation'],
                        snr_range=(10.0, 30.0),
                        doppler_range=(0.0, 500.0),
                        channel_profiles=['TDL_B', 'TDL_C', 'TDL_D', 'SIMPLE'],
                        pilot_configs=['1_pilot_A', '1_pilot_B', '2_pilots_A', '2_pilots_B'],
                        device=self.device
                    )

                    self.train_loader = DataLoader(
                        phase2_dataset, batch_size=cfg['batch_size'],
                        shuffle=True, num_workers=cfg['num_workers'], drop_last=True
                    )

                    for pg in self.optimizer.param_groups:
                        pg['lr'] = 1e-4

                    self.scheduler = WarmupCosineScheduler(
                        self.optimizer, max_lr=1e-4, min_lr=1e-6,
                        total_steps=(cfg['total_steps'] - step), warmup_steps=100
                    )
                    self.scheduler.current_step = 0

                    break

                loss, ber = self._train_step(batch_data)
                lr = self.scheduler.step()

                running_loss += loss
                running_ber += ber
                running_count += 1
                step += 1

                if step % cfg['log_every'] == 0:
                    avg_loss = running_loss / running_count
                    avg_ber = running_ber / running_count
                    elapsed = time.time() - train_start
                    speed = step / max(elapsed, 1)
                    eta = (cfg['total_steps'] - step) / max(speed, 0.01)

                    self.history['train_loss'].append(avg_loss)
                    self.history['train_ber'].append(avg_ber)
                    self.history['learning_rates'].append(lr)
                    self.history['steps'].append(step)

                    print(f"  Step {step:>6}/{cfg['total_steps']} | Loss: {avg_loss:.4f} | "
                          f"BER: {avg_ber:.4f} | LR: {lr:.2e} | ETA: {eta/60:.1f}min")

                    running_loss = 0.0
                    running_ber = 0.0
                    running_count = 0

                if step % cfg['val_every'] == 0:
                    val = self._validate()
                    self.history['val_loss'].append(val['loss'])
                    self.history['val_ber'].append(val['ber_viterbi'])
                    self.history['val_steps'].append(step)

                    print(f"\n  {'─'*60}")
                    print(f"  Validation at Step {step}")
                    print(f"    Validation Loss:  {val['loss']:.4f}")
                    print(f"    Validation BER:   {val['ber_viterbi']:.6f}")

                    if val['ber_viterbi'] < self.best_val_ber:
                        self.best_val_ber = val['ber_viterbi']
                        self._save_checkpoint(step, is_best=True)
                        print(f"    ★ New best model saved!")

                    print(f"  {'─'*60}\n")

        self._save_checkpoint(step, is_best=False)
        self.plot_metrics()

        total_time = time.time() - train_start
        print(f"\n{'='*70}")
        print(f"  Training Complete!")
        print(f"  Total time:    {total_time/60:.1f} minutes")
        print(f"  Best Val BER:  {self.best_val_ber:.6f}")
        print(f"{'='*70}\n")

        return self.history

    def _train_step(self, batch_data):
        self.model.train()
        inputs = batch_data['input'].to(self.device)
        targets = batch_data['target_bits'].to(self.device)
        data_mask = batch_data['data_mask'].to(self.device)
        bit_mask = batch_data['bit_mask'].to(self.device).unsqueeze(-1).unsqueeze(-1)
        snr_map = batch_data['snr_map'].to(self.device)

        snr_weights = None
        if 'snr' in batch_data:
            raw_snr = batch_data['snr'].to(self.device)
            snr_weights = torch.clamp(torch.exp(raw_snr / 7.5), min=1.0, max=25.0)

        logits = self.model(inputs, snr_map=snr_map, phase=Phase.TRAIN, data_mask=data_mask)
        loss = self.criterion(logits, targets, data_mask, bit_mask, snr_weights)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        self.optimizer.step()

        with torch.no_grad():
            ber = compute_ber(logits, targets, data_mask, bit_mask)
        return loss.item(), ber

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, total_ber, n = 0, 0, 0
        max_batches = 30

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_loader):
                if i >= max_batches:
                    break

                inputs = batch_data['input'].to(self.device)
                targets = batch_data['target_bits'].to(self.device)
                data_mask = batch_data['data_mask'].to(self.device)
                bit_mask = batch_data['bit_mask'].to(self.device).unsqueeze(-1).unsqueeze(-1)
                snr_map = batch_data['snr_map'].to(self.device)

                logits_for_loss = self.model(inputs, snr_map=snr_map, phase=Phase.TRAIN, data_mask=data_mask)
                detected_bits = self.model(inputs, snr_map=snr_map, phase=Phase.TEST, data_mask=data_mask)

                loss = self.criterion(logits_for_loss, targets, data_mask, bit_mask)
                ber = compute_ber(detected_bits, targets, data_mask, bit_mask)

                total_loss += loss.item()
                total_ber += ber
                n += 1

        return {
            'loss': total_loss / max(n, 1),
            'ber_viterbi': total_ber / max(n, 1)
        }

    def _save_checkpoint(self, step, is_best=False):
        name = 'best_model.pt' if is_best else f'checkpoint_step{step}.pt'
        path = os.path.join(self.save_dir, name)

        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

        torch.save({
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ber': self.best_val_ber,
            'config': self.config
        }, path)

        hist_path = os.path.join(self.save_dir, 'history.json')
        serializable = {k: [float(x) for x in v] for k, v in self.history.items()}
        with open(hist_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def plot_metrics(self):
        print("\n  Generating training plots...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        steps = self.history['steps']
        val_steps = self.history['val_steps']

        axes[0].plot(steps, self.history['train_loss'], label='Train Loss', alpha=0.8, color='blue')
        if self.history['val_loss']:
            axes[0].plot(val_steps, self.history['val_loss'], label='Val Loss', marker='o', color='red')
        axes[0].set_title('BCE Loss vs. Training Steps')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        axes[1].plot(steps, self.history['train_ber'], label='Train BER', alpha=0.8, color='blue')
        if self.history['val_ber']:
            axes[1].plot(val_steps, self.history['val_ber'], label='Val BER (Trellis)', marker='o', color='red')
        axes[1].set_title('Bit Error Rate (BER) vs. Training Steps')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('BER')
        axes[1].set_yscale('log')
        axes[1].grid(True, linestyle='--', alpha=0.6, which='both')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  ★ Plots saved successfully to: {plot_path}")


if __name__ == "__main__":
    Nr = 2
    B_max = 8
    modulation = '16QAM'

    config = {
        'max_lr': 1e-3,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'warmup_steps': 500,
        'batch_size': 16,
        'total_steps': 40000,
        'phase2_step': 25000,
        'val_every': 500,
        'log_every': 50,
        'grad_clip': 1.0,
        'num_workers': 0,
        'modulation': modulation,
        'save_dir': 'checkpoints',
        'experiment_name': 'viterbinet_deeprx',
        'seed': 42,
    }

    print("\nCreating training dataset (10,000 samples)...")
    train_dataset = DeepRxDataset(
        n_samples=10000, n_rx_antennas=Nr, modulation=modulation,
        snr_range=(-4.0, 30.0), doppler_range=(0.0, 500.0),
        channel_profiles=['TDL_B', 'TDL_C', 'TDL_D', 'SIMPLE'],
        pilot_configs=['1_pilot_A', '1_pilot_B', '2_pilots_A', '2_pilots_B'],
        device=DEVICE
    )

    print("Creating validation dataset (2,000 samples)...")
    val_dataset = DeepRxDataset(
        n_samples=2000, n_rx_antennas=Nr, modulation=modulation,
        snr_range=(0.0, 25.0), doppler_range=(10.0, 300.0),
        channel_profiles=['TDL_A', 'SIMPLE'],
        pilot_configs=['2_pilots_A'],
        device=DEVICE
    )

    print("\nInitializing ViterbiNet Detector...")
    in_features = 2 * (2 * Nr + 1) + 1
    n_states = 2 ** MEMORY_LENGTH
    model = ViterbiNetDetector(n_states=n_states, in_features=in_features, max_bits=B_max)

    trainer = ViterbiNetTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=DEVICE
    )

    history = trainer.train()