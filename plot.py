"""
============================================================================
DeepRx vs NeuroAdaRX: Publication-Quality Plots with Confidence Intervals
============================================================================
"""

from traditional_receiver import TraditionalReceiver
from ofdm_system import (
    QAMModulator, OFDMTransmitter, OFDMReceiver,
    ChannelModel, add_awgn
)
from deeprx_model import (
    DeepRx, build_deeprx_input, compute_ber,
    create_pilot_mask, generate_qpsk_pilots, create_bit_mask
)
from model import ViterbiNetDetector, MEMORY_LENGTH, Phase 

import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import argparse

import matplotlib
matplotlib.use('Agg')


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

STYLE = {
    'gnn':    {'color': '#8E24AA', 'marker': 'D', 'linewidth': 2.5,
               'markersize': 8, 'label': 'NeuroAdaRX'},
    'deeprx': {'color': '#1565C0', 'marker': 'o', 'linewidth': 2.5,
               'markersize': 8, 'label': 'DeepRx (ResNet Baseline)'},
    'lmmse':  {'color': '#C62828', 'marker': 's', 'linewidth': 2.5,
               'markersize': 8, 'label': 'LMMSE (Classical)', 'linestyle': '--'},
}

FIG_DIR = 'figures'

def setup_style():
    plt.rcParams.update({
        'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 16,
        'legend.fontsize': 12, 'figure.figsize': (8, 6), 'figure.dpi': 150,
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2,
        'font.family': 'serif', 'savefig.format': 'pdf',
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: GNN Prediction Wrapper
# ═══════════════════════════════════════════════════════════════════════════

def get_gnn_predictions(model_gnn, data):
    """
    Executes the GNN forward pass with the correct Context Normalization 
    and handles the deep supervision dimension.
    """
    # Pass Phase.TEST to trigger the Trellis Decoding
    detected_bits = model_gnn(
        rx=data['input'], 
        snr_map=data['snr_map'], 
        phase=Phase.TEST, 
        data_mask=data['data_mask']
    )
    
    # If model returns deep supervision iterations [..., max_iters], extract the final one
    if detected_bits.dim() == 5:
        detected_bits = detected_bits[..., -1]
        
    return detected_bits


# ═══════════════════════════════════════════════════════════════════════════
#  Test Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_test_batch(batch_size, snr_db, doppler_hz,
                        channel_profile='TDL_B', pilot_config='2_pilots_A',
                        Nr=2, S=14, F=312, n_fft=512, cp_length=36,
                        device='cpu'):
    modulation = '16QAM'
    from deeprx_model import MODULATION_CONFIG
    bps = MODULATION_CONFIG[modulation]

    pilot_mask = create_pilot_mask(S, F, pilot_config, device)
    data_mask = 1.0 - pilot_mask
    bit_mask = create_bit_mask(modulation, 8, device)
    n_data = int(data_mask.sum().item())

    tx = OFDMTransmitter(F, n_fft, cp_length, S)
    rx_fe = OFDMReceiver(F, n_fft, cp_length, S)

    data_bits, data_syms = QAMModulator.bits_to_symbols(
        batch_size * n_data, modulation, device)
    data_syms = data_syms.reshape(batch_size, n_data)
    data_bits = data_bits.reshape(batch_size, n_data, -1)

    pilot_symbols = generate_qpsk_pilots(batch_size, S, F, pilot_mask, device)
    grid, target_bits, _ = tx.build_resource_grid(
        data_syms, pilot_symbols, pilot_mask, data_bits, bps)
    tx_signal = tx.modulate_ofdm(grid)
    sig_len = tx_signal.shape[1]
    sig_power = (tx_signal.abs() ** 2).mean().item()

    rx_waveforms = []
    for ant in range(Nr):
        ch = ChannelModel(channel_profile, doppler_hz, device=device)
        h_time, _ = ch.generate(batch_size, sig_len, S, n_fft, cp_length)
        rx_ant = ch.apply_channel(tx_signal, h_time)
        rx_ant = add_awgn(rx_ant, snr_db, sig_power)
        rx_waveforms.append(rx_ant)

    rx_multi = torch.stack(rx_waveforms, dim=1)
    rx_grid = rx_fe.demodulate(rx_multi, Nr)
    deeprx_input = build_deeprx_input(rx_grid, pilot_symbols)

    # EXACT REPLICATION of your dataset logic: normalized snr_map
    snr_map = torch.full((batch_size, 1, S, F), snr_db / 30.0, dtype=torch.float32, device=device)

    return {
        'input': deeprx_input, 
        'target_bits': target_bits,
        'data_mask': data_mask, 
        'bit_mask': bit_mask,
        'rx_grid': rx_grid, 
        'tx_pilots': pilot_symbols,
        'pilot_mask': pilot_mask,
        'snr_map': snr_map 
    }


def evaluate_with_confidence(model_deeprx, model_gnn, param_list, param_name,
                             fixed_params, n_trials=20,
                             batch_size=8, device='cpu'):
    trad_rx = TraditionalReceiver('16QAM', device)
    model_deeprx.eval()
    model_gnn.eval()

    results = {
        'params': param_list,
        'deeprx_mean': [], 'deeprx_ci': [],
        'gnn_mean': [], 'gnn_ci': [],
        'lmmse_mean': [], 'lmmse_ci': [],
    }

    for param_val in param_list:
        ber_deep_trials = []
        ber_gnn_trials = []
        ber_lmmse_trials = []

        for trial in range(n_trials):
            if param_name == 'snr':
                data = generate_test_batch(
                    batch_size, param_val, fixed_params.get('doppler', 50.0),
                    fixed_params.get('channel', 'TDL_B'),
                    fixed_params.get('pilot', '2_pilots_A'),
                    device=device)
            elif param_name == 'doppler':
                data = generate_test_batch(
                    batch_size, fixed_params.get('snr', 15.0), param_val,
                    fixed_params.get('channel', 'TDL_B'),
                    fixed_params.get('pilot', '2_pilots_A'),
                    device=device)

            with torch.no_grad():
                logits_d = model_deeprx(data['input'])
                logits_g = get_gnn_predictions(model_gnn, data)
            
            bd = compute_ber(logits_d, data['target_bits'], data['data_mask'], data['bit_mask'])
            bg = compute_ber(logits_g, data['target_bits'], data['data_mask'], data['bit_mask'])

            llrs = trad_rx.process(data['rx_grid'], data['tx_pilots'], data['pilot_mask'])
            bl = compute_ber(llrs, data['target_bits'], data['data_mask'], data['bit_mask'])

            ber_deep_trials.append(bd)
            ber_gnn_trials.append(bg)
            ber_lmmse_trials.append(bl)

        deep_arr = np.array(ber_deep_trials)
        gnn_arr = np.array(ber_gnn_trials)
        lmmse_arr = np.array(ber_lmmse_trials)

        ci_factor = 1.96  

        results['deeprx_mean'].append(deep_arr.mean())
        results['deeprx_ci'].append(ci_factor * deep_arr.std() / np.sqrt(n_trials))
        
        results['gnn_mean'].append(gnn_arr.mean())
        results['gnn_ci'].append(ci_factor * gnn_arr.std() / np.sqrt(n_trials))

        results['lmmse_mean'].append(lmmse_arr.mean())
        results['lmmse_ci'].append(ci_factor * lmmse_arr.std() / np.sqrt(n_trials))

        gain = deep_arr.mean() / max(gnn_arr.mean(), 1e-8)
        print(f"    {param_name}={param_val:>6.1f} | "
              f"GNN: {gnn_arr.mean():.4f} | DeepRx: {deep_arr.mean():.4f} | "
              f"LMMSE: {lmmse_arr.mean():.4f} | GNN Gain over DeepRx: {gain:.2f}x")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 1: BER vs SNR with Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def plot_ber_vs_snr(model_deeprx, model_gnn, device='cpu'):
    print("\n  [1/6] BER vs SNR with 95% CI...")
    snr_list = list(range(0, 27, 3))
    results = evaluate_with_confidence(
        model_deeprx, model_gnn, snr_list, 'snr',
        {'doppler': 50.0, 'channel': 'TDL_B', 'pilot': '2_pilots_A'},
        n_trials=25, batch_size=8, device=device
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    dm = np.array(results['deeprx_mean'])
    dc = np.array(results['deeprx_ci'])
    gm = np.array(results['gnn_mean'])
    gc = np.array(results['gnn_ci'])
    lm = np.array(results['lmmse_mean'])
    lc = np.array(results['lmmse_ci'])

    ax.semilogy(snr_list, gm, **STYLE['gnn'])
    ax.fill_between(snr_list, np.maximum(gm-gc, 1e-5), gm+gc, alpha=0.15, color=STYLE['gnn']['color'])

    ax.semilogy(snr_list, dm, **STYLE['deeprx'])
    ax.fill_between(snr_list, np.maximum(dm-dc, 1e-5), dm+dc, alpha=0.15, color=STYLE['deeprx']['color'])

    ax.semilogy(snr_list, lm, **STYLE['lmmse'])
    ax.fill_between(snr_list, np.maximum(lm-lc, 1e-5), lm+lc, alpha=0.15, color=STYLE['lmmse']['color'])

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Uncoded BER')
    ax.set_title('BER vs SNR — GNN vs DeepRx vs LMMSE\n(16-QAM, Doppler=50 Hz, TDL-B, 2 Pilots, shaded=95% CI)')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xlim([snr_list[0], snr_list[-1]])
    ax.set_ylim([1e-4, 1])
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_ber_vs_snr.pdf'))
    plt.close()
    print(f"    Saved: fig1_ber_vs_snr.pdf")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 2: BER vs Doppler with Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def plot_ber_vs_doppler(model_deeprx, model_gnn, device='cpu'):
    print("\n  [2/6] BER vs Doppler with 95% CI...")
    doppler_list = [10, 50, 100, 150, 200, 300, 400, 500]
    results = evaluate_with_confidence(
        model_deeprx, model_gnn, doppler_list, 'doppler',
        {'snr': 15.0, 'channel': 'TDL_B', 'pilot': '2_pilots_A'},
        n_trials=25, batch_size=8, device=device
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    dm = np.array(results['deeprx_mean'])
    dc = np.array(results['deeprx_ci'])
    gm = np.array(results['gnn_mean'])
    gc = np.array(results['gnn_ci'])
    lm = np.array(results['lmmse_mean'])
    lc = np.array(results['lmmse_ci'])

    ax.semilogy(doppler_list, gm, **STYLE['gnn'])
    ax.fill_between(doppler_list, np.maximum(gm-gc, 1e-5), gm+gc, alpha=0.15, color=STYLE['gnn']['color'])

    ax.semilogy(doppler_list, dm, **STYLE['deeprx'])
    ax.fill_between(doppler_list, np.maximum(dm-dc, 1e-5), dm+dc, alpha=0.15, color=STYLE['deeprx']['color'])

    ax.semilogy(doppler_list, lm, **STYLE['lmmse'])
    ax.fill_between(doppler_list, np.maximum(lm-lc, 1e-5), lm+lc, alpha=0.15, color=STYLE['lmmse']['color'])

    ax.set_xlabel('Maximum Doppler Shift (Hz)')
    ax.set_ylabel('Uncoded BER')
    ax.set_title('BER vs Doppler Shift — GNN vs DeepRx vs LMMSE\n(16-QAM, SNR=15 dB, TDL-B, 2 Pilots, shaded=95% CI)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_ber_vs_doppler.pdf'))
    plt.close()
    print(f"    Saved: fig2_ber_vs_doppler.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 3: Per-Bit BER
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_bit_ber(model_deeprx, model_gnn, device='cpu'):
    print("\n  [3/6] Per-Bit BER Analysis...")
    trad_rx = TraditionalReceiver('16QAM', device)
    bit_names = ['I-MSB\n($b_0$)', 'I-LSB\n($b_1$)', 'Q-MSB\n($b_2$)', 'Q-LSB\n($b_3$)']

    ber_deep, ber_gnn, ber_lmmse = [], [], []
    ci_deep, ci_gnn, ci_lmmse = [], [], []

    for bit_idx in range(4):
        single_mask = torch.zeros(1, 8, 1, 1, device=device)
        single_mask[0, bit_idx, 0, 0] = 1.0

        bd_trials, bg_trials, bl_trials = [], [], []
        for _ in range(30):
            data = generate_test_batch(8, 15.0, 50.0, device=device)
            with torch.no_grad():
                logits_d = model_deeprx(data['input'])
                logits_g = get_gnn_predictions(model_gnn, data)
            
            bd_trials.append(compute_ber(logits_d, data['target_bits'], data['data_mask'], single_mask))
            bg_trials.append(compute_ber(logits_g, data['target_bits'], data['data_mask'], single_mask))
            
            llrs = trad_rx.process(data['rx_grid'], data['tx_pilots'], data['pilot_mask'])
            bl_trials.append(compute_ber(llrs, data['target_bits'], data['data_mask'], single_mask))

        bd_arr, bg_arr, bl_arr = np.array(bd_trials), np.array(bg_trials), np.array(bl_trials)
        
        ber_deep.append(bd_arr.mean())
        ber_gnn.append(bg_arr.mean())
        ber_lmmse.append(bl_arr.mean())
        
        ci_deep.append(1.96 * bd_arr.std() / np.sqrt(30))
        ci_gnn.append(1.96 * bg_arr.std() / np.sqrt(30))
        ci_lmmse.append(1.96 * bl_arr.std() / np.sqrt(30))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.25

    ax.bar(x - width, ber_gnn, width, yerr=ci_gnn, color=STYLE['gnn']['color'], label='NeuroAdaRX', alpha=0.85, capsize=4)
    ax.bar(x, ber_deep, width, yerr=ci_deep, color=STYLE['deeprx']['color'], label='DeepRx', alpha=0.85, capsize=4)
    ax.bar(x + width, ber_lmmse, width, yerr=ci_lmmse, color=STYLE['lmmse']['color'], label='LMMSE', alpha=0.85, capsize=4)

    ax.set_ylabel('BER')
    ax.set_title('Per-Bit BER — 16-QAM\n(SNR=15 dB, Doppler=50 Hz, error bars=95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(bit_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_per_bit_ber.pdf'))
    plt.close()
    print(f"    Saved: fig3_per_bit_ber.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 4: Training History
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_history(deeprx_path, gnn_path):
    print("\n  [4/6] Training History (Combined)...")
    def load_hist(path):
        dir_path = os.path.dirname(path)
        hist_file = os.path.join(dir_path, 'history.json')
        if os.path.exists(hist_file):
            with open(hist_file) as f:
                return json.load(f)
        return None

    h_deeprx = load_hist(deeprx_path)
    h_gnn = load_hist(gnn_path)

    if not h_deeprx and not h_gnn:
        print("    No history.json found in either model directory. Skipping Plot 4.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    def plot_val_line(h_data, label, style_key):
        if h_data and h_data.get('val_ber'):
            n_val = len(h_data['val_ber'])
            val_interval = h_data['steps'][-1] // n_val if n_val > 0 else 500
            val_steps = list(range(val_interval, val_interval * n_val + 1, val_interval))[:n_val]
            ax.plot(val_steps, h_data['val_ber'], color=STYLE[style_key]['color'],
                    marker=STYLE[style_key]['marker'], linewidth=2.5, markersize=8, label=label)

    plot_val_line(h_gnn, 'Val BER (NeuroAdaRX)', 'gnn')
    plot_val_line(h_deeprx, 'Val BER (DeepRx)', 'deeprx')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation BER')
    ax.set_title('Validation BER Progression Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_training_history.pdf'))
    plt.close()
    print(f"    Saved: fig4_training_history.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 5: Channel Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_channel_comparison(model_deeprx, model_gnn, device='cpu'):
    print("\n  [5/6] Channel Model Comparison...")
    trad_rx = TraditionalReceiver('16QAM', device)
    channels = ['TDL_A', 'TDL_B', 'TDL_C', 'TDL_D', 'SIMPLE']
    labels = ['TDL-A\n(NLOS)', 'TDL-B\n(NLOS)', 'TDL-C\n(NLOS)', 'TDL-D\n(LOS)', 'Simple\n(NLOS)']

    ber_deep, ber_gnn, ber_lmmse = [], [], []
    ci_d, ci_g, ci_l = [], [], []

    for ch in channels:
        bd_trials, bg_trials, bl_trials = [], [], []
        for _ in range(20):
            data = generate_test_batch(8, 15.0, 100.0, channel_profile=ch, device=device)
            with torch.no_grad():
                logits_d = model_deeprx(data['input'])
                logits_g = get_gnn_predictions(model_gnn, data)
            
            bd_trials.append(compute_ber(logits_d, data['target_bits'], data['data_mask'], data['bit_mask']))
            bg_trials.append(compute_ber(logits_g, data['target_bits'], data['data_mask'], data['bit_mask']))
            
            llrs = trad_rx.process(data['rx_grid'], data['tx_pilots'], data['pilot_mask'])
            bl_trials.append(compute_ber(llrs, data['target_bits'], data['data_mask'], data['bit_mask']))

        bd_arr, bg_arr, bl_arr = np.array(bd_trials), np.array(bg_trials), np.array(bl_trials)
        
        ber_deep.append(bd_arr.mean())
        ber_gnn.append(bg_arr.mean())
        ber_lmmse.append(bl_arr.mean())
        
        ci_d.append(1.96 * bd_arr.std() / np.sqrt(20))
        ci_g.append(1.96 * bg_arr.std() / np.sqrt(20))
        ci_l.append(1.96 * bl_arr.std() / np.sqrt(20))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(channels))
    width = 0.25

    ax.bar(x - width, ber_gnn, width, yerr=ci_g, color=STYLE['gnn']['color'], label='NeuroAdaRX', alpha=0.85, capsize=4)
    ax.bar(x, ber_deep, width, yerr=ci_d, color=STYLE['deeprx']['color'], label='DeepRx', alpha=0.85, capsize=4)
    ax.bar(x + width, ber_lmmse, width, yerr=ci_l, color=STYLE['lmmse']['color'], label='LMMSE', alpha=0.85, capsize=4)

    ax.set_ylabel('BER')
    ax.set_title('BER by Channel Model\n(SNR=15 dB, Doppler=100 Hz, 16-QAM, error bars=95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_channel_comparison.pdf'))
    plt.close()
    print(f"    Saved: fig5_channel_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 6: Gain Summary (GNN vs DeepRx)
# ═══════════════════════════════════════════════════════════════════════════

def plot_gain_summary(snr_results):
    print("\n  [6/6] Performance Gain Summary (GNN vs DeepRx)...")
    snr_list = snr_results['params']
    dm = np.array(snr_results['deeprx_mean'])
    gm = np.array(snr_results['gnn_mean'])
    
    gains = dm / np.maximum(gm, 1e-8)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#8E24AA' if g > 1 else '#1565C0' for g in gains]
    bars = ax.bar(snr_list, gains, width=2.0, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Equal Performance')
    ax.text(snr_list[1], max(gains)*0.92, 'NeuroAdaRX Better ↑', fontsize=12, color='#8E24AA', fontweight='bold')
    ax.text(snr_list[1], 0.15, 'DeepRx Better ↓', fontsize=12, color='#1565C0', fontweight='bold')

    for bar, g in zip(bars, gains):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.02,
                f'{g:.2f}×', ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Gain (BER$_{DeepRx}$ / BER$_{GNN}$)')
    ax.set_title('NeuroAdaRX Multiplier Gain over DeepRx ResNet')
    ax.legend(loc='upper right')
    ax.set_xlim([snr_list[0]-2, snr_list[-1]+2])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_gain_summary.pdf'))
    plt.close()
    print(f"    Saved: fig6_gain_summary.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots comparing DeepRx and NeuroAdaRX.")
    parser.add_argument('--deeprx_path', type=str, 
                        default='deeprx.pt', 
                        help="Exact file path to deeprx.pt weights")
    parser.add_argument('--gnn_path', type=str, 
                        default='gnn.pt', 
                        help="Exact file path to gnn.pt weights")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"{'DeepRx vs NeuroAdaRX — Evaluation Suite':^70}")
    print("=" * 70)

    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # Load DeepRx
    print(f"  Loading DeepRx from: {args.deeprx_path}")
    model_deeprx = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8)
    ckpt_d = torch.load(args.deeprx_path, map_location=device, weights_only=False)
    model_deeprx.load_state_dict(ckpt_d['model_state_dict'] if 'model_state_dict' in ckpt_d else ckpt_d)
    model_deeprx = model_deeprx.to(device)
    model_deeprx.eval()

    # Load NeuroAdaRX
    print(f"  Loading NeuroAdaRX from: {args.gnn_path}")
    
    Nr = 2
    in_features = 2 * (2 * Nr + 1) + 1 
    n_states = 2 ** MEMORY_LENGTH
    
    model_gnn = ViterbiNetDetector(n_states=n_states, in_features=in_features, max_bits=8) 
    ckpt_g = torch.load(args.gnn_path, map_location=device, weights_only=False)
    
    state_dict_g = ckpt_g['model_state_dict'] if 'model_state_dict' in ckpt_g else ckpt_g
    clean_state_dict_g = {}
    for k, v in state_dict_g.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict_g[name] = v
        
    model_gnn.load_state_dict(clean_state_dict_g)
    model_gnn = model_gnn.to(device)
    model_gnn.eval()

    # Generate all plots
    snr_results = plot_ber_vs_snr(model_deeprx, model_gnn, device)
    plot_ber_vs_doppler(model_deeprx, model_gnn, device)
    plot_per_bit_ber(model_deeprx, model_gnn, device)
    plot_training_history(args.deeprx_path, args.gnn_path)
    plot_channel_comparison(model_deeprx, model_gnn, device)
    plot_gain_summary(snr_results)

    print(f"\n{'='*70}")
    print(f"  All 6 figures saved to: {FIG_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()