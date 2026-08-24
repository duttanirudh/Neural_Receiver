import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import from your modules
from model import ViterbiNetDetector, MEMORY_LENGTH, DEVICE
from data_generator import DeepRxDataset
from train import ViterbiNetTrainer


if __name__ == "__main__":
    Nr = 2
    B_max = 8
    modulation = '16QAM'

    config = {
        'max_lr': 1e-3,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'warmup_steps': 500,
        'batch_size': 8,  # Kept at 8 to prevent CUDA OOM with 4-Layer ResNet + 64-Dim GNN
        
        # --- NEW: Two-Stage Pipeline Controls ---
        'total_steps': 40000,          # Total training budget
        'phase2_step': 9999999,          # When the dataset shifts to High-SNR (10dB - 30dB)
        
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
    # +1 added for the explicitly concatenated snr_map tensor
    in_features = 2 * (2 * Nr + 1) + 1 
    n_states = 2 ** MEMORY_LENGTH
    
    # 1. Initialize the base model
    model = ViterbiNetDetector(n_states=n_states, in_features=in_features, max_bits=B_max)
    
    # 2. Check for multiple GPUs and wrap the model if they exist
    if torch.cuda.device_count() > 1:
        print(f"  --> Utilizing {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        
    # 3. Move the model to the primary device
    model = model.to(DEVICE)

    print("\nInitializing ViterbiNet DeepRx Trainer...")
    trainer = ViterbiNetTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=DEVICE
    )

    print("Starting training and evaluation loop...")
    history = trainer.train()