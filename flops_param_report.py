"""
============================================================================
DeepRx-GNN vs DeepRx — Parameter Count, FLOPs, Memory Footprint,
Frontend Architecture Comparison, and NRX-style Iterations-vs-BER Plot
============================================================================

METHODOLOGY NOTES (read before citing any number from this script)
--------------------------------------------------------------------------

1. PARAMETER COUNTS are exact: sum(p.numel() for p in model.parameters()).
   For the GNN, split into "frontend" (everything in ViterbiNetDetector
   NOT under the `gnn` submodule) and "gnn" (TrellisMessagePassing) by
   filtering `model.named_parameters()` on the `gnn.` prefix. This is
   robust to future frontend changes (it doesn't hardcode a block list),
   as long as the message-passing module stays named `self.gnn`.

2. FLOPs use a HYBRID method, not a single library, because of a real gap
   in common FLOP-profiling tools:
     - Frontend (all Conv2d/BatchNorm2d) and DeepRx (also pure conv):
       profiled with `thop`, which has solid built-in support for these
       layer types. To isolate the frontend cleanly WITHOUT re-implementing
       ViterbiNetDetector.forward()'s layer sequence by hand, the script
       temporarily swaps `model.gnn` for a shape-preserving no-op stub,
       profiles the REAL model.forward() end-to-end (through a keyword-
       safe bridge module — see _KwargForwardBridge — so a future change
       to forward()'s positional argument order can't silently mis-bind
       arguments during profiling), then restores the real gnn. The
       frontend is never re-implemented anywhere in this file.
     - GNN (TrellisMessagePassing): computed ANALYTICALLY in closed form
       from module dimensions (embed/msg_forward/msg_backward/node_update/
       decode), NOT via thop. Reason: thop's default hook table frequently
       has no entry for nn.GRUCell, silently contributing 0 FLOPs for the
       single most expensive op in the GNN's per-iteration cost. This
       script separately asks thop for its (possibly wrong) GNN-only
       number and prints it next to the analytical number so any gap is
       visible and citable in the methods section.
     - MAC->FLOP convention: FLOPs = 2 x MACs (multiply + add), the
       standard convention thop itself uses. Stated explicitly in every
       table so units are unambiguous.
     - FLOPs are computed at EVERY GNN depth (1..max_iterations), not just
       the two endpoints, so they can be joined directly against an ACT
       threshold->depth table from a separate latency/BER script.

3. FRONTEND-SIZE COMPARISON TABLE is built by LIVE introspection of
   `named_modules()` on both loaded models — not by repeating architecture
   descriptions from prior handoff notes, which may no longer match the
   current deeprx_model.py.

4. MEMORY FOOTPRINT reports THREE numbers, not one:
     - parameter bytes
     - buffer bytes (e.g. BatchNorm running stats — real memory traffic,
       not "parameters", easy to under-count if skipped)
     - PEAK ACTIVATION memory during one real forward pass, measured
       directly via torch.cuda.max_memory_allocated (not estimated from
       architecture). This is the number that actually matters for
       deployment memory footprint, and static weight bytes alone will
       understate it, sometimes by a lot.
   Peak activation memory is measured as an INCREMENT above whatever is
   already resident on the GPU when the forward pass runs — run this in a
   fresh process/session (not appended after building TensorRT engines in
   the same session) or the number will be contaminated.

5. FAIRNESS NOTE ON THE DeepRx/GNN RATIO: comparing DeepRx against the
   GNN's max-iterations FLOPs overstates the GNN's cost relative to how it
   actually runs under ACT (which typically halts well below max_iters).
   This script reports the ratio AT EVERY DEPTH, not just at max_iters, so
   the summary table can cite the ratio at the ACT-selected depth rather
   than a worst-case endpoint. This project already had one FLOP/param-
   based efficiency claim ("1/3 the computational cost") get falsified by
   a real latency measurement — see the caveat printed at the end of this
   script's BER-vs-iterations plot section. Don't repeat that mistake by
   picking whichever endpoint looks best.

6. The "FLOPs per unit of BER improvement" number in the final plot section
   is DELIBERATELY presented as a single-operating-point snapshot, clearly
   caveated at print time, not a general efficiency curve.

7. Everything that needs REAL trained weights (only the final NRX-style
   BER-vs-iterations plot) loads a checkpoint using the same priority list
   as eval_act.py/plot.py. Parameter/FLOP/memory counts are architecture
   properties and are correct with or without trained weights, so those
   sections run even if no checkpoint is found (clearly flagged).

8. metadata (batch size used, device, whether real checkpoints were
   loaded) is saved INSIDE the JSON report itself — every number here is
   batch-size-dependent, and without this recorded, a report read back
   later has no way to know what it's normalized against.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ViterbiNetDetector, Phase, compute_ber, MEMORY_LENGTH

try:
    from deeprx_model import DeepRx
    HAVE_DEEPRX = True
except ImportError as e:
    HAVE_DEEPRX = False
    print(f"[WARNING] Could not import DeepRx from deeprx_model.py: {e}")
    print("          DeepRx-side numbers will be skipped.")

try:
    from plot import generate_test_batch
    HAVE_TEST_BATCH = True
except ImportError as e:
    HAVE_TEST_BATCH = False
    print(f"[WARNING] Could not import generate_test_batch from plot.py: {e}")
    print("          Falling back to torch.randn dummy inputs for FLOP counting "
          "(fine for FLOPs/params/memory, NOT valid for the BER-vs-iterations plot).")

try:
    from thop import profile as thop_profile
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False
    print("[WARNING] thop not installed. Run: pip install thop --break-system-packages")
    print("          Frontend/DeepRx FLOPs will be skipped; params/memory still reported.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = 'figures_flops'
CI_FACTOR = 1.96


# ═══════════════════════════════════════════════════════════════════════════
#  Analytical FLOP formulas (Linear, GRUCell) — no library dependency
# ═══════════════════════════════════════════════════════════════════════════

def linear_flops(N, in_features, out_features):
    """Exact. N = number of positions/tokens the layer is applied to."""
    macs = N * in_features * out_features
    matmul_flops = 2 * macs           # multiply + add
    bias_flops = N * out_features     # bias add
    return matmul_flops + bias_flops


def gru_cell_flops(N, input_size, hidden_size):
    """
    Exact for the matmul terms (the dominant cost); the elementwise gate
    term (sigmoid x2, tanh x1, plus the (1-z)*n + z*h combination) is a
    standard order-of-magnitude approximation, flagged as such — it is
    consistently <5% of total GRUCell cost for hidden sizes in this range,
    the matmuls dominate.
    """
    macs = 3 * hidden_size * (input_size + hidden_size) * N
    matmul_flops = 2 * macs
    bias_flops = N * 6 * hidden_size  # bias_ih (3h) + bias_hh (3h)
    elementwise_flops_approx = N * hidden_size * 14  # APPROX, see docstring
    return matmul_flops + bias_flops + elementwise_flops_approx, elementwise_flops_approx


# ═══════════════════════════════════════════════════════════════════════════
#  Param counting
# ═══════════════════════════════════════════════════════════════════════════

def count_params_split(model):
    frontend_params = 0
    gnn_params = 0
    for name, p in model.named_parameters():
        if name.startswith('gnn.'):
            gnn_params += p.numel()
        else:
            frontend_params += p.numel()
    total = frontend_params + gnn_params
    return {'frontend': frontend_params, 'gnn': gnn_params, 'total': total}


def memory_footprint_mb(model):
    """Static weight/buffer memory only. See peak_activation_memory_mb()
    for the activation-memory number that belongs alongside this."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        'param_mb': param_bytes / 1e6,
        'buffer_mb': buffer_bytes / 1e6,
        'total_mb': (param_bytes + buffer_bytes) / 1e6,
    }


def peak_activation_memory_mb(forward_fn, device=DEVICE):
    """forward_fn: zero-arg callable that runs exactly one forward pass.
    Measures INCREMENTAL peak memory above whatever's already resident on
    the device (weights, buffers, any other model loaded in the same
    session) — isolates activation memory specifically. Returns None on
    CPU (no CUDA memory stats to read).
    CAUTION: run this in a fresh process, or immediately after loading
    just the model(s) being measured — if this runs after building
    TensorRT engines or loading a second model in the same session, the
    'already resident' baseline includes that extra memory and the
    subtraction will be wrong.
    """
    if device.type != 'cuda':
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        forward_fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated(device)
    return (peak_alloc - base_alloc) / 1e6


# ═══════════════════════════════════════════════════════════════════════════
#  Live architecture introspection (no hardcoded assumptions)
# ═══════════════════════════════════════════════════════════════════════════

def describe_conv_stack(model, exclude_prefix=None):
    """Enumerates every nn.Conv2d in named_modules() order, live, from
    whatever model is actually loaded — not from prior handoff notes."""
    rows = []
    for name, m in model.named_modules():
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        if isinstance(m, nn.Conv2d):
            rows.append({
                'name': name,
                'in_channels': m.in_channels,
                'out_channels': m.out_channels,
                'kernel_size': m.kernel_size,
                'dilation': m.dilation,
                'stride': m.stride,
            })
    return rows


def print_conv_stack_table(rows, title):
    print(f"\n  {title}")
    print(f"  {'layer':<20} {'in_ch':>7} {'out_ch':>7} {'kernel':>10} {'dilation':>10}")
    for r in rows:
        print(f"  {r['name']:<20} {r['in_channels']:>7} {r['out_channels']:>7} "
              f"{str(r['kernel_size']):>10} {str(r['dilation']):>10}")
    if rows:
        depth = len(rows)
        peak_ch = max(r['out_channels'] for r in rows)
        print(f"  -> depth = {depth} conv layers, peak channel width = {peak_ch}")


# ═══════════════════════════════════════════════════════════════════════════
#  Shape-preserving stub — isolates frontend FLOPs without re-implementing
#  ViterbiNetDetector.forward()'s layer sequence anywhere in this file.
# ═══════════════════════════════════════════════════════════════════════════

class ShapePreservingGNNStub(nn.Module):
    """Drop-in replacement for TrellisMessagePassing during frontend-only
    profiling. embed:n_states->hidden and decode:hidden->n_states in the
    real module net out to an identity shape transform (n_states in,
    n_states out), so returning the input unchanged preserves the exact
    tensor shape ViterbiNetDetector.forward() expects downstream, without
    computing anything or duplicating any real layer code."""
    def __init__(self):
        super().__init__()
        self.last_iters_executed = 0

    def forward(self, node_features, data_mask=None, bit_mask=None,
                halt_threshold=None, max_iterations_override=None, max_bits=None):
        return node_features


class _KwargForwardBridge(nn.Module):
    """thop calls the wrapped module positionally: model(*inputs). Binding
    a raw ViterbiNetDetector positionally is fragile — if forward()'s
    argument order ever changes, thop would silently mis-bind arguments
    and produce a plausible-looking but WRONG FLOPs number with no error.
    This bridge fixes the call to explicit keywords so profiling can't
    silently drift out of sync with the real signature."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rx, snr_map, data_mask, bit_mask):
        return self.model(rx=rx, snr_map=snr_map, phase=Phase.TEST,
                           data_mask=data_mask, bit_mask=bit_mask)


# ═══════════════════════════════════════════════════════════════════════════
#  FLOP counting: frontend (thop, via stub-swap) + GNN (analytical)
# ═══════════════════════════════════════════════════════════════════════════

def profile_frontend_flops_thop(model, sample_batch):
    """Swaps in the shape-preserving stub, profiles the REAL forward()
    (through the keyword-safe bridge) with thop, restores the real gnn.
    Returns (MACs, FLOPs, thop_reported_params)."""
    if not HAVE_THOP:
        return None
    original_gnn = model.gnn
    model.gnn = ShapePreservingGNNStub()
    bridge = _KwargForwardBridge(model)
    try:
        macs, params = thop_profile(
            bridge,
            inputs=(sample_batch['input'], sample_batch['snr_map'],
                    sample_batch['data_mask'], sample_batch['bit_mask']),
            verbose=False,
        )
    finally:
        model.gnn = original_gnn
    return {'macs': macs, 'flops': 2 * macs, 'thop_params': params}


def profile_gnn_flops_thop_for_comparison(model, N_positions, n_states, hidden_dim):
    """thop's OWN estimate of the GNN alone, for cross-checking against the
    analytical numbers below. Expected to under-report if thop's version
    lacks a GRUCell hook — that gap is the point of running this."""
    if not HAVE_THOP:
        return None
    gnn = model.gnn
    dummy = torch.randn(N_positions, n_states, device=next(gnn.parameters()).device)
    original_max_iter = gnn.max_iterations
    gnn.max_iterations = 1
    try:
        macs, params = thop_profile(gnn, inputs=(dummy.unsqueeze(0) if dummy.dim() == 2 else dummy,), verbose=False)
    except Exception as e:
        print(f"  [thop] GNN profiling failed outright ({e}) — see analytical numbers below.")
        macs, params = None, None
    finally:
        gnn.max_iterations = original_max_iter
    if macs is None:
        return None
    return {'macs': macs, 'flops': 2 * macs, 'thop_params': params}


def compute_gnn_flops_analytical(gnn, N_positions, n_iterations):
    """Closed-form GNN FLOPs for `n_iterations` message-passing steps, plus
    the fixed embed+decode cost that runs once per forward() call
    regardless of iteration count (see model.py: embed before the loop,
    decode after it)."""
    n_states = gnn.embed.in_features
    hidden = gnn.hidden_dim

    embed_flops = linear_flops(N_positions, gnn.embed.in_features, gnn.embed.out_features)
    decode_flops = linear_flops(N_positions, gnn.decode.in_features, gnn.decode.out_features)
    fixed_overhead = embed_flops + decode_flops

    msg_fwd_flops = linear_flops(N_positions, gnn.msg_forward.in_features, gnn.msg_forward.out_features)
    msg_bwd_flops = linear_flops(N_positions, gnn.msg_backward.in_features, gnn.msg_backward.out_features)
    gru_flops, gru_elementwise_approx = gru_cell_flops(
        N_positions, gnn.node_update.input_size, gnn.node_update.hidden_size)

    per_iteration_flops = msg_fwd_flops + msg_bwd_flops + gru_flops
    per_iteration_linear_only = msg_fwd_flops + msg_bwd_flops  # for thop cross-check (no GRUCell)

    total_flops = fixed_overhead + n_iterations * per_iteration_flops

    return {
        'n_states': n_states, 'hidden_dim': hidden,
        'embed_flops': embed_flops, 'decode_flops': decode_flops,
        'fixed_overhead_flops': fixed_overhead,
        'msg_forward_flops': msg_fwd_flops, 'msg_backward_flops': msg_bwd_flops,
        'gru_cell_flops': gru_flops, 'gru_elementwise_approx_flops': gru_elementwise_approx,
        'per_iteration_flops': per_iteration_flops,
        'per_iteration_linear_only_flops': per_iteration_linear_only,
        'n_iterations': n_iterations,
        'total_flops': total_flops,
    }


def compute_gnn_flops_all_depths(gnn, N_positions, max_iterations):
    """FLOPs at EVERY depth 1..max_iterations, not just the two endpoints —
    so this can be joined directly against an ACT threshold->depth table
    from a separate latency/BER script (e.g. threshold 0.99 -> mean_iters
    4.48 -> nearest depth 4 -> look up depth 4 here)."""
    return {k: compute_gnn_flops_analytical(gnn, N_positions, k)
            for k in range(1, max_iterations + 1)}


# ═══════════════════════════════════════════════════════════════════════════
#  Summary table — one flat row per model, ready for the frozen paper table
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_table(report, gnn_params, gnn_mem, deeprx_total_params=None,
                         deeprx_mem=None, act_depth_for_gnn=None):
    """One flat row per model. `act_depth_for_gnn`: pass the ACT-selected
    depth (rounded mean_iters from a separate eager-mode ACT sweep) so the
    GNN's FLOPs are reported at its REAL deployed cost, not a worst-case
    endpoint — see fairness note in the module docstring. If not provided,
    falls back to max_iters and says so explicitly in `flops_basis`, so
    nobody reading the summary mistakes it for an ACT-representative
    number by accident."""
    depth_used = act_depth_for_gnn
    depth_note = 'ACT-selected depth (from eager sweep)' if act_depth_for_gnn is not None \
        else 'max_iters — NOT ACT-representative, pass act_depth_for_gnn once the eager sweep is available'
    if depth_used is None:
        depth_used = max(report['gnn_flops_by_depth'].keys())

    frontend_flops = report.get('frontend_flops_thop', {}).get('flops', 0) or 0
    gnn_flops_total = frontend_flops + report['gnn_flops_by_depth'][depth_used]['total_flops']

    rows = {
        'DeepRx-GNN': {
            'n_params_total': gnn_params['total'],
            'weight_memory_mb': gnn_mem['total_mb'],
            'peak_activation_memory_mb': report.get('gnn_peak_activation_mb'),
            'flops_per_inference': gnn_flops_total,
            'flops_basis': f'{depth_note}, depth={depth_used}',
        },
    }
    if deeprx_total_params is not None:
        rows['DeepRx'] = {
            'n_params_total': deeprx_total_params,
            'weight_memory_mb': deeprx_mem['total_mb'],
            'peak_activation_memory_mb': report.get('deeprx_peak_activation_mb'),
            'flops_per_inference': report.get('deeprx_flops_thop', {}).get('flops'),
            'flops_basis': 'single fixed-depth pass',
        }
    return rows


# ═══════════════════════════════════════════════════════════════════════════
#  Main report
# ═══════════════════════════════════════════════════════════════════════════

def load_gnn(device):
    Nr = 2
    in_features = 2 * (2 * Nr + 1) + 1
    n_states = 2 ** MEMORY_LENGTH
    model = ViterbiNetDetector(n_states=n_states, in_features=in_features, max_bits=8).to(device)

    ckpt_paths = ["best_model.pt", "gnn_with_act.pt", "gnn.pt"]
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            state_dict = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
            print(f"Loaded GNN checkpoint: {ckpt_path}")
            return model, True
    print("[WARNING] No GNN checkpoint found — using random init. "
          "Params/FLOPs/memory are still exact (architecture-only); the "
          "BER-vs-iterations plot at the end will be skipped.")
    return model, False


def load_deeprx(device, path='deeprx.pt'):
    if not HAVE_DEEPRX:
        return None, False
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=8).to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        print(f"Loaded DeepRx checkpoint: {path}")
        return model, True
    print(f"[WARNING] DeepRx checkpoint '{path}' not found — using random init. "
          "Params/FLOPs/memory are still exact.")
    return model, False


def get_sample_batch(batch_size, device):
    if HAVE_TEST_BATCH:
        return generate_test_batch(batch_size, 12.0, 50.0, device=device)
    # Fallback dummy — shapes must match model expectations; only used if
    # plot.py's real data pipeline isn't importable. Fine for FLOPs/params/
    # memory (architecture-only), NOT valid for the BER-vs-iterations plot.
    S, F, Nr = 14, 312, 2
    in_features = 2 * (2 * Nr + 1) + 1
    return {
        'input': torch.randn(batch_size, in_features - 1, S, F, device=device),
        'snr_map': torch.full((batch_size, 1, S, F), 0.4, device=device),
        'data_mask': torch.ones(1, 1, S, F, device=device),
        'bit_mask': torch.ones(1, 8, 1, 1, device=device),
        'target_bits': torch.randint(0, 2, (batch_size, 8, S, F), device=device).float(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deeprx_ckpt', type=str, default='deeprx.pt')
    parser.add_argument('--flop_batch_size', type=int, default=1,
                        help="Batch size for FLOP/param dummy input (1 = standard per-sample convention).")
    parser.add_argument('--act_depth_for_gnn', type=int, default=None,
                        help="ACT-selected depth (rounded mean_iters from a separate eager-mode "
                             "ACT sweep) to use for the summary table's GNN FLOPs/ratio. If "
                             "omitted, the summary falls back to max_iters and flags this "
                             "explicitly — fill this in once you have the eager sweep results.")
    parser.add_argument('--ber_plot_snrs', type=float, nargs='+', default=[6.0, 24.0],
                        help="Representative SNRs for the NRX-style iters-vs-BER plot.")
    parser.add_argument('--ber_plot_trials', type=int, default=25)
    parser.add_argument('--ber_plot_batch_size', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Using device: {DEVICE}\n")

    gnn_model, gnn_trained = load_gnn(DEVICE)
    deeprx_model, deeprx_trained = load_deeprx(DEVICE, args.deeprx_ckpt)
    gnn_model.eval()
    if deeprx_model is not None:
        deeprx_model.eval()

    sample = get_sample_batch(args.flop_batch_size, DEVICE)

    report = {
        'metadata': {
            'flop_batch_size': args.flop_batch_size,
            'device': str(DEVICE),
            'gnn_checkpoint_loaded': gnn_trained,
            'deeprx_checkpoint_loaded': deeprx_trained,
            'have_thop': HAVE_THOP,
            'have_real_test_batch': HAVE_TEST_BATCH,
        }
    }

    # ── 1. Parameter counts ────────────────────────────────────────────
    print("=" * 78)
    print("1. PARAMETER COUNTS")
    print("=" * 78)
    gnn_params = count_params_split(gnn_model)
    print(f"  DeepRx-GNN — frontend: {gnn_params['frontend']:>10,} | "
          f"gnn: {gnn_params['gnn']:>10,} | total: {gnn_params['total']:>10,}")
    report['gnn_params'] = gnn_params

    deeprx_total_params = None
    if deeprx_model is not None:
        deeprx_total_params = sum(p.numel() for p in deeprx_model.parameters())
        print(f"  DeepRx (monolithic)  — total: {deeprx_total_params:>10,}  "
              f"(no natural frontend/backend split — single ResNet stack)")
        print(f"\n  Ratio (DeepRx / GNN total params): {deeprx_total_params / gnn_params['total']:.2f}x")
        report['deeprx_params'] = deeprx_total_params
    else:
        print("  DeepRx not available — skipped.")

    # ── 2. Memory footprint (weights+buffers, THEN peak activation) ───
    print("\n" + "=" * 78)
    print("2. MEMORY FOOTPRINT")
    print("=" * 78)
    gnn_mem = memory_footprint_mb(gnn_model)
    print(f"  DeepRx-GNN weight memory — params: {gnn_mem['param_mb']:.3f} MB | "
          f"buffers (BN stats): {gnn_mem['buffer_mb']:.3f} MB | "
          f"total: {gnn_mem['total_mb']:.3f} MB")
    report['gnn_memory_mb'] = gnn_mem

    gnn_peak_mb = peak_activation_memory_mb(
        lambda: gnn_model(rx=sample['input'], snr_map=sample['snr_map'], phase=Phase.TEST,
                           data_mask=sample['data_mask'], bit_mask=sample['bit_mask']),
        device=DEVICE)
    report['gnn_peak_activation_mb'] = gnn_peak_mb
    print(f"  DeepRx-GNN peak activation memory (batch={args.flop_batch_size}): "
          + (f"{gnn_peak_mb:.3f} MB" if gnn_peak_mb is not None else "N/A (CPU device)"))

    deeprx_mem = None
    if deeprx_model is not None:
        deeprx_mem = memory_footprint_mb(deeprx_model)
        print(f"\n  DeepRx weight memory     — params: {deeprx_mem['param_mb']:.3f} MB | "
              f"buffers (BN stats): {deeprx_mem['buffer_mb']:.3f} MB | "
              f"total: {deeprx_mem['total_mb']:.3f} MB")
        report['deeprx_memory_mb'] = deeprx_mem
        print(f"  Ratio (DeepRx / GNN total weight memory): "
              f"{deeprx_mem['total_mb'] / gnn_mem['total_mb']:.2f}x")

        deeprx_peak_mb = peak_activation_memory_mb(
            lambda: deeprx_model(sample['input']), device=DEVICE)
        report['deeprx_peak_activation_mb'] = deeprx_peak_mb
        print(f"  DeepRx peak activation memory (batch={args.flop_batch_size}): "
              + (f"{deeprx_peak_mb:.3f} MB" if deeprx_peak_mb is not None else "N/A (CPU device)"))

        print("\n  [memory-wall note] Weight memory alone is not deployment memory — peak "
              "activation memory is measured directly above, not estimated. If this script "
              "runs after other models/engines were loaded in the same session, the peak-"
              "activation numbers are contaminated by that prior memory; run in a fresh "
              "process for a clean number.")

    # ── 3. Frontend architecture comparison (live introspection) ──────
    print("\n" + "=" * 78)
    print("3. FRONTEND ARCHITECTURE — LIVE INTROSPECTION (not from prior handoff notes)")
    print("=" * 78)
    gnn_conv_rows = describe_conv_stack(gnn_model, exclude_prefix='gnn')
    print_conv_stack_table(gnn_conv_rows, "DeepRx-GNN frontend (CNN stack, excludes TrellisMessagePassing):")
    report['gnn_frontend_convs'] = gnn_conv_rows

    if deeprx_model is not None:
        deeprx_conv_rows = describe_conv_stack(deeprx_model)
        print_conv_stack_table(deeprx_conv_rows, "DeepRx (whole model — monolithic ResNet):")
        report['deeprx_convs'] = deeprx_conv_rows

    # ── 4. FLOPs ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("4. FLOPs (FLOPs = 2 x MACs, thop convention)")
    print("=" * 78)

    frontend_flop_result = profile_frontend_flops_thop(gnn_model, sample)
    if frontend_flop_result:
        print(f"  Frontend (thop, isolated via stub-swap on real forward()): "
              f"{frontend_flop_result['flops']:,.0f} FLOPs "
              f"({frontend_flop_result['macs']:,.0f} MACs) per {args.flop_batch_size}-sample batch")
        print(f"  Frontend param count per thop: {frontend_flop_result['thop_params']:,.0f} "
              f"(cross-check vs. named_parameters split: {gnn_params['frontend']:,})")
        report['frontend_flops_thop'] = frontend_flop_result
    else:
        print("  Frontend FLOPs skipped (thop unavailable).")

    S, n_sc = sample['data_mask'].shape[2], sample['data_mask'].shape[3]
    N_positions = args.flop_batch_size * gnn_model.max_bits * S * n_sc
    print(f"\n  GNN token count for this batch: N = batch({args.flop_batch_size}) x "
          f"max_bits({gnn_model.max_bits}) x S({S}) x F({n_sc}) = {N_positions:,}")

    max_iters = gnn_model.gnn.max_iterations
    gnn_flops_by_depth = compute_gnn_flops_all_depths(gnn_model.gnn, N_positions, max_iters)
    gnn_flops_analytical = gnn_flops_by_depth[max_iters]
    gnn_flops_1iter = gnn_flops_by_depth[1]
    report['gnn_flops_by_depth'] = gnn_flops_by_depth
    report['gnn_flops_analytical_full'] = gnn_flops_analytical    # kept for backward compat
    report['gnn_flops_analytical_1iter'] = gnn_flops_1iter        # kept for backward compat

    print(f"\n  GNN (analytical, exact for Linear layers, GRUCell elementwise term "
          f"approximate — see docstring):")
    print(f"    embed (once/call):       {gnn_flops_analytical['embed_flops']:>16,.0f} FLOPs")
    print(f"    decode (once/call):      {gnn_flops_analytical['decode_flops']:>16,.0f} FLOPs")
    print(f"    msg_forward (per iter):  {gnn_flops_analytical['msg_forward_flops']:>16,.0f} FLOPs")
    print(f"    msg_backward (per iter): {gnn_flops_analytical['msg_backward_flops']:>16,.0f} FLOPs")
    print(f"    GRUCell (per iter):      {gnn_flops_analytical['gru_cell_flops']:>16,.0f} FLOPs "
          f"(of which ~{gnn_flops_analytical['gru_elementwise_approx_flops']:,.0f} is the "
          f"approximate elementwise-gate term)")
    print(f"    -> per-iteration total:  {gnn_flops_analytical['per_iteration_flops']:>16,.0f} FLOPs")

    print(f"\n  GNN FLOPs by depth (join this against an ACT threshold->depth table by depth):")
    for k, r in gnn_flops_by_depth.items():
        print(f"    depth={k}: {r['total_flops']:>16,.0f} FLOPs")

    gnn_thop_check = profile_gnn_flops_thop_for_comparison(
        gnn_model, N_positions, gnn_flops_analytical['n_states'], gnn_flops_analytical['hidden_dim'])
    if gnn_thop_check:
        analytical_1iter_linear_only = (gnn_flops_1iter['fixed_overhead_flops']
                                        + gnn_flops_1iter['per_iteration_linear_only_flops'])
        print(f"\n  [cross-check] thop's own GNN estimate (1 iter): "
              f"{gnn_thop_check['flops']:,.0f} FLOPs")
        print(f"  [cross-check] analytical Linear-only estimate (1 iter, excludes GRUCell): "
              f"{analytical_1iter_linear_only:,.0f} FLOPs")
        gap = gnn_flops_1iter['total_flops'] - gnn_thop_check['flops']
        pct = 100 * gap / gnn_flops_1iter['total_flops']
        print(f"  [cross-check] full analytical (incl. GRUCell) vs thop: "
              f"{gap:,.0f} FLOPs missing from thop ({pct:.1f}% of true total) "
              f"-- {'consistent with thop lacking a GRUCell hook' if pct > 5 else 'small gap, thop may support GRUCell in this version'}")
        report['gnn_thop_cross_check'] = gnn_thop_check

    deeprx_flop_result = None
    if deeprx_model is not None and HAVE_THOP:
        # [FIX] previously gated on HAVE_TEST_BATCH too, which needlessly
        # skipped DeepRx FLOPs whenever plot.py wasn't importable — FLOPs
        # only need correct shapes (get_sample_batch's fallback provides
        # those), not real signal data.
        macs, params = thop_profile(deeprx_model, inputs=(sample['input'],), verbose=False)
        deeprx_flop_result = {'macs': macs, 'flops': 2 * macs, 'thop_params': params}
        print(f"\n  DeepRx total (thop): {deeprx_flop_result['flops']:,.0f} FLOPs "
              f"({deeprx_flop_result['macs']:,.0f} MACs) per {args.flop_batch_size}-sample batch")
        report['deeprx_flops_thop'] = deeprx_flop_result
    elif deeprx_model is not None:
        print("\n  DeepRx FLOPs skipped (thop unavailable).")

    if deeprx_flop_result and frontend_flop_result:
        print(f"\n  DeepRx / GNN FLOPs ratio, BY DEPTH (fixed frontend cost + depth-dependent "
              f"GNN cost) — do not cite only the max_iters endpoint, see fairness note in the "
              f"module docstring:")
        ratio_by_depth = {}
        for k, r in gnn_flops_by_depth.items():
            total_k = frontend_flop_result['flops'] + r['total_flops']
            ratio = deeprx_flop_result['flops'] / total_k
            ratio_by_depth[k] = {'total_gnn_flops': total_k, 'ratio_deeprx_over_gnn': ratio}
            print(f"    depth={k}: total GNN FLOPs={total_k:,.0f} | DeepRx/GNN ratio={ratio:.2f}x")
        report['flops_ratio_by_depth'] = ratio_by_depth
        report['total_gnn_flops_max_iters'] = frontend_flop_result['flops'] + gnn_flops_analytical['total_flops']
        report['total_deeprx_flops'] = deeprx_flop_result['flops']

    # ── 5. Summary table — one flat row per model ──────────────────────
    print("\n" + "=" * 78)
    print("5. SUMMARY — ready for the frozen comparison table")
    print("=" * 78)
    summary = build_summary_table(
        report, gnn_params, gnn_mem, deeprx_total_params, deeprx_mem,
        act_depth_for_gnn=args.act_depth_for_gnn)
    report['summary_table'] = summary
    if args.act_depth_for_gnn is None:
        print("  [NOTE] --act_depth_for_gnn was not passed — GNN FLOPs below are at max_iters, "
              "which OVERSTATES real ACT-deployed cost. Re-run with --act_depth_for_gnn set to "
              "round(mean_iters) from your eager ACT sweep once available, for the real number.")
    for name, row in summary.items():
        peak_str = f"{row['peak_activation_memory_mb']:.3f}" if row['peak_activation_memory_mb'] is not None else "N/A"
        flops_str = f"{row['flops_per_inference']:,.0f}" if row['flops_per_inference'] is not None else "N/A"
        print(f"  {name}: params={row['n_params_total']:,} | "
              f"weight_mem={row['weight_memory_mb']:.3f}MB | "
              f"peak_activation_mem={peak_str}MB | "
              f"FLOPs/inference={flops_str} ({row['flops_basis']})")

    # ── 6. Save raw report ──────────────────────────────────────────────
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    json_path = os.path.join(args.out_dir, 'flops_params_report.json')
    with open(json_path, 'w') as f:
        json.dump(_clean(report), f, indent=2)
    print(f"\nSaved raw numbers: {json_path}")

    # ── 7. NRX-style: BER vs iterations, FLOPs overlaid ────────────────
    if gnn_trained and HAVE_TEST_BATCH:
        print("\n" + "=" * 78)
        print("6. NRX-style plot: BER vs iterations, FLOPs overlaid")
        print("=" * 78)
        plot_ber_and_flops_vs_iterations(
            gnn_model, args.ber_plot_snrs, N_positions_per_iter_fn=lambda bs: bs * gnn_model.max_bits * S * n_sc,
            n_trials=args.ber_plot_trials, batch_size=args.ber_plot_batch_size,
            out_dir=args.out_dir)
    else:
        print("\n[skipped] NRX-style BER-vs-iterations plot needs a trained GNN "
              "checkpoint and generate_test_batch — one or both unavailable.")

    print(f"\nDone. Report + figures in: {args.out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
#  NRX Fig-4-style: BLER/BER vs iterations, with FLOPs (and, secondarily,
#  a single-operating-point "FLOPs per unit BER improvement" framing)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ber_and_flops_vs_iterations(model, snr_list, N_positions_per_iter_fn,
                                      n_trials=25, batch_size=8, out_dir=OUT_DIR):
    model.eval()
    max_iters = model.gnn.max_iterations
    iter_range = list(range(1, max_iters + 1))
    N_positions = N_positions_per_iter_fn(batch_size)

    fig, axes = plt.subplots(1, len(snr_list), figsize=(6.5 * len(snr_list), 5.5), sharey=False)
    if len(snr_list) == 1:
        axes = [axes]

    all_rows = []
    for ax, snr in zip(axes, snr_list):
        ber_means, ber_cis, flops_list = [], [], []
        for k in iter_range:
            bers = []
            for _ in range(n_trials):
                data = generate_test_batch(batch_size, snr, 50.0, device=DEVICE)
                with torch.no_grad():
                    preds = model(rx=data['input'], snr_map=data['snr_map'], phase=Phase.TEST,
                                  data_mask=data['data_mask'], bit_mask=data['bit_mask'],
                                  max_iterations_override=k)
                bers.append(compute_ber(preds, data['target_bits'], data['data_mask'], data['bit_mask']))
            bers = np.array(bers)
            ber_means.append(bers.mean())
            ber_cis.append(CI_FACTOR * bers.std() / np.sqrt(n_trials))
            flops_k = compute_gnn_flops_analytical(model.gnn, N_positions, k)['total_flops']
            flops_list.append(flops_k)
            all_rows.append({'snr_db': snr, 'iterations': k, 'ber_mean': float(bers.mean()),
                             'ber_ci95': float(CI_FACTOR * bers.std() / np.sqrt(n_trials)),
                             'gnn_flops': float(flops_k)})
            print(f"  SNR={snr:>5.1f} dB | iters={k} | BER={bers.mean():.5f} | "
                  f"GNN FLOPs={flops_k:,.0f}")

        ber_means = np.array(ber_means)
        ber_cis = np.array(ber_cis)

        color_ber = '#8E24AA'
        ax.semilogy(iter_range, ber_means, color=color_ber, marker='D', linewidth=2.2,
                   markersize=7, label='BER (left axis)')
        ax.fill_between(iter_range, np.maximum(ber_means - ber_cis, 1e-6), ber_means + ber_cis,
                        alpha=0.15, color=color_ber)
        ax.set_xlabel('Iterations executed')
        ax.set_ylabel('Uncoded BER', color=color_ber)
        ax.tick_params(axis='y', labelcolor=color_ber)
        ax.set_title(f'SNR = {snr:.0f} dB')
        ax.set_xticks(iter_range)
        ax.grid(True, which='both', alpha=0.3)

        ax2 = ax.twinx()
        color_flops = '#1565C0'
        ax2.plot(iter_range, flops_list, color=color_flops, marker='s', linestyle='--',
                linewidth=1.8, markersize=6, label='GNN FLOPs (right axis)')
        ax2.set_ylabel('GNN FLOPs per forward pass', color=color_flops)
        ax2.tick_params(axis='y', labelcolor=color_flops)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    fig.suptitle('DeepRx-GNN: BER vs Iterations Executed, FLOPs Overlaid\n'
                '(NRX-style depth/complexity tradeoff view — cf. NRX Fig. 4)',
                fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, 'fig_flops1_ber_vs_iterations.pdf')
    plt.savefig(path)
    plt.close()
    print(f"\n  Saved: {path}")

    csv_path = os.path.join(out_dir, 'ber_vs_iterations_flops.csv')
    with open(csv_path, 'w') as f:
        f.write('snr_db,iterations,ber_mean,ber_ci95,gnn_flops\n')
        for row in all_rows:
            f.write(f"{row['snr_db']},{row['iterations']},{row['ber_mean']},"
                    f"{row['ber_ci95']},{row['gnn_flops']}\n")
    print(f"  Saved: {csv_path}")

    print("\n  [CAVEAT — read before using this as an 'efficiency' number in the paper]")
    print("  The numbers above are FLOPs, not latency. This project already had one")
    print("  FLOP/param-based efficiency claim ('1/3 the computational cost') get")
    print("  falsified by a real CUDA-event latency measurement (GNN turned out 5.4x")
    print("  SLOWER at inference, because sequential GRU iterations don't parallelize")
    print("  the way stacked independent conv layers do). Any 'FLOPs per dB of BER")
    print("  improvement' framing in the paper should be shown ALONGSIDE the measured")
    print("  latency number from the ACT uniform-protocol sweep, not as a substitute")
    print("  for it — present it as a single-operating-point snapshot, not a general")
    print("  efficiency curve.")


if __name__ == '__main__':
    main()