#new model
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODULATION_CONFIG = {'QPSK': 2, '16QAM': 4, '64QAM': 6, '256QAM': 8}

HIDDEN1_SIZE = 75
HIDDEN2_SIZE = 16

HALF = 0.5
MEMORY_LENGTH = 4


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'

def create_pilot_mask(S=14, F=312, config='2_pilots_A', device='cpu'):
    mask = torch.zeros(1, 1, S, F, device=device)
    if config == '1_pilot_A':
        mask[0, 0, 2, 0::2] = 1.0
    elif config == '1_pilot_B':
        mask[0, 0, 2, 1::2] = 1.0
    elif config == '2_pilots_A':
        mask[0, 0, 2, 0::2] = 1.0
        mask[0, 0, 11, 1::2] = 1.0
    elif config == '2_pilots_B':
        mask[0, 0, 2, 1::2] = 1.0
        mask[0, 0, 11, 0::2] = 1.0
    else:
        raise ValueError(f"Unknown config: {config}")
    return mask

def generate_qpsk_pilots(batch, S, F, pilot_mask, device='cpu'):
    signs_r = 2 * torch.randint(0, 2, (batch, 1, S, F), device=device).float() - 1
    signs_i = 2 * torch.randint(0, 2, (batch, 1, S, F), device=device).float() - 1
    qpsk = (1.0 / math.sqrt(2)) * torch.complex(signs_r, signs_i)
    return qpsk * pilot_mask

def compute_ber(predictions, target_bits, data_mask, bit_mask):
    full_mask = (data_mask * bit_mask).expand_as(target_bits)
    if predictions.dim() == 5:
        pred_states = torch.argmax(predictions, dim=-1)
        pred = (pred_states % 2).float()
    else:
        pred = predictions.float()
        
    errors = (pred != target_bits).float()
    n_errors = (errors * full_mask).sum()
    n_total = full_mask.sum().clamp(min=1.0)
    return (n_errors / n_total).item()

def create_bit_mask(modulation, B=8, device='cpu'):
    n_bits = MODULATION_CONFIG[modulation]
    mask = torch.zeros(1, B, 1, 1, device=device)
    mask[0, :n_bits, 0, 0] = 1.0
    return mask

def build_deeprx_input(rx_signal, tx_pilots):
    Nr = rx_signal.shape[1]
    tx_pilots_expanded = tx_pilots.expand(-1, Nr, -1, -1)
    raw_ch_est = rx_signal * torch.conj(tx_pilots_expanded)
    Z_complex = torch.cat([rx_signal, tx_pilots, raw_ch_est], dim=1)
    Z = torch.cat([Z_complex.real, Z_complex.imag], dim=1)
    return Z

def compute_target_states(target_bits: torch.Tensor, memory_length: int) -> torch.Tensor:
    batch_size, B, S, F = target_bits.shape
    flat_bits = target_bits.permute(0, 1, 2, 3).reshape(batch_size * B, S * F)
    
    states = torch.zeros_like(flat_bits, dtype=torch.long)
    for i in range(memory_length):
        shifted = torch.roll(flat_bits, shifts=i, dims=1)
        shifted[:, :i] = 0  
        states += (shifted * (2 ** i)).long()
        
    return states.reshape(batch_size, B, S, F)


class DeepRxLoss(nn.Module):
    def __init__(self, memory_length=MEMORY_LENGTH):
        super().__init__()
        self.memory_length = memory_length
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, target_bits, data_mask, bit_mask, snr_weights=None):
        target_states = compute_target_states(target_bits, self.memory_length)
        full_mask = (data_mask * bit_mask).expand_as(target_bits)
        
        logits_ce = logits.permute(0, 4, 1, 2, 3) 
        ce_loss = self.ce(logits_ce, target_states)
        
        # --- NEW: Dynamic SNR Penalty Weighting ---
        if snr_weights is not None:
            snr_weights = snr_weights.view(-1, 1, 1, 1)
            ce_loss = ce_loss * snr_weights
            
        masked_loss = (ce_loss * full_mask).sum() / full_mask.sum().clamp(min=1.0)
        return masked_loss

class ResNetBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation  # keep spatial size constant: padding = dilation for 3x3 kernel
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise 3x3 (per-channel) + pointwise 1x1 (channel mixing).
    Same receptive field as Conv2d(channels, channels, k=3), at roughly
    channels/9 the parameter count for large channel counts."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=padding,
            dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise 3x3 (per-channel) + pointwise 1x1 (channel mixing).
    Same receptive field as Conv2d(channels, channels, k=3), at roughly
    channels/9 the parameter count for large channel counts."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=padding,
            dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableResNetBlock(nn.Module):
    """Drop-in replacement for ResNetBlock — identical interface and
    dilation/channel behavior, only the conv type inside changes."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(channels, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = DepthwiseSeparableConv2d(channels, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class TrellisMessagePassing(nn.Module):
    def __init__(self, n_states: int, hidden_dim: int = 64, max_iterations: int = 12):
        super().__init__()
        self.n_states = n_states
        self.max_iterations = max_iterations
        self.hidden_dim = hidden_dim

        self.embed = nn.Linear(n_states, hidden_dim)
        self.msg_forward = nn.Linear(hidden_dim, hidden_dim)
        self.msg_backward = nn.Linear(hidden_dim, hidden_dim)
        self.node_update = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)
        self.decode = nn.Linear(hidden_dim, n_states)

        self.halt_eval = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    
    def forward(self, node_features: torch.Tensor, data_mask: torch.Tensor = None,
                bit_mask: torch.Tensor = None, halt_threshold: float = None,
                max_iterations_override: int = None, max_bits: int = None) -> torch.Tensor:
        batch_seq, seq_len, _ = node_features.shape
        node_states = self.embed(node_features)

        n_iters_to_run = max_iterations_override if max_iterations_override is not None else self.max_iterations

        full_mask = None
        S = n_sc = batch_size_dim = None
        if data_mask is not None and bit_mask is not None and max_bits is not None:
            batch_size_dim = batch_seq // max_bits          # FIX: derived from real max_bits, not data_mask.shape[0]
            S, n_sc = data_mask.shape[2], data_mask.shape[3]
            full_mask = (data_mask * bit_mask).expand(batch_size_dim, max_bits, S, n_sc)

        iters_executed = n_iters_to_run
        for i in range(n_iters_to_run):
            m_fwd = self.msg_forward(node_states)
            m_bwd = self.msg_backward(node_states)

            m_fwd_shifted = torch.roll(m_fwd, shifts=1, dims=1)
            m_fwd_shifted[:, 0, :] = 0
            m_bwd_shifted = torch.roll(m_bwd, shifts=-1, dims=1)
            m_bwd_shifted[:, -1, :] = 0

            aggregated_msgs = torch.cat([m_fwd_shifted, m_bwd_shifted], dim=-1)
            flat_msgs = aggregated_msgs.view(-1, self.hidden_dim * 2)
            flat_nodes = node_states.view(-1, self.hidden_dim)

            updated_flat_nodes = self.node_update(flat_msgs, flat_nodes)
            node_states = updated_flat_nodes.view(batch_seq, seq_len, self.hidden_dim)

            # ACT halting is now OPT-IN: only fires if halt_threshold is explicitly passed.
            # (Previously this fired automatically at a hardcoded 0.997 whenever both
            # masks were present — see caveat below.)
            if not self.training and full_mask is not None and halt_threshold is not None:
                halt_grid = self.halt_eval(node_states).view(batch_size_dim, max_bits, S, n_sc)
                payload_confidence = ((halt_grid * full_mask).sum() / full_mask.sum().clamp(min=1.0)).item()
                if payload_confidence > halt_threshold:
                    iters_executed = i + 1
                    break

        self.last_iters_executed = iters_executed   # always set — no separate instrumentation needed
        return self.decode(node_states)
    
    def forward_with_snapshots(self, node_features: torch.Tensor, data_mask: torch.Tensor = None):
        """
        Diagnostic-only: runs the same message-passing loop as forward(), but
        returns the decoded output at EVERY iteration instead of just the last.
        Used to profile how many iterations are actually needed for predictions
        to stabilize, before committing to any adaptive-halting retrain.
        Does not affect forward() or any existing eval/training path.
        """
        batch_seq, seq_len, _ = node_features.shape
        node_states = self.embed(node_features)

        snapshots = []
        for i in range(self.max_iterations):
            m_fwd = self.msg_forward(node_states)
            m_bwd = self.msg_backward(node_states)

            m_fwd_shifted = torch.roll(m_fwd, shifts=1, dims=1)
            m_fwd_shifted[:, 0, :] = 0
            m_bwd_shifted = torch.roll(m_bwd, shifts=-1, dims=1)
            m_bwd_shifted[:, -1, :] = 0

            aggregated_msgs = torch.cat([m_fwd_shifted, m_bwd_shifted], dim=-1)
            flat_msgs = aggregated_msgs.view(-1, self.hidden_dim * 2)
            flat_nodes = node_states.view(-1, self.hidden_dim)

            updated_flat_nodes = self.node_update(flat_msgs, flat_nodes)
            node_states = updated_flat_nodes.view(batch_seq, seq_len, self.hidden_dim)

            # Decode THIS iteration's state — same decode head used at the end
            # of the normal forward() pass, just called every iteration here.
            iter_output = self.decode(node_states)
            snapshots.append(iter_output)

        return snapshots  # list of length max_iterations, each (batch_seq, seq_len, n_states)
    
class ViterbiNetDetector(nn.Module):
    def __init__(self, n_states: int, in_features: int = 11, max_bits: int = 8):
        super(ViterbiNetDetector, self).__init__()
        self.n_states = n_states
        self.in_features = in_features
        self.max_bits = max_bits

        # --- EVEN MORE WIDENED FRONTEND (depthwise-separable) ---
        # Ramping: 64 -> 128 -> 256 channels (matches DeepRx's peak width)
        # Blocks now use depthwise-separable convs instead of standard Conv2d,
        # targeting DeepRx-level parameter efficiency at the same channel/depth
        # schedule. conv_in/conv_up1/conv_up2/conv_out unchanged — the bulk of
        # the parameter cost was in the two 3x3 convs per block, not the 1x1s.
        self.conv_in = nn.Conv2d(self.in_features, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)

        self.block1 = DepthwiseSeparableResNetBlock(64, dilation=1)
        self.block2 = DepthwiseSeparableResNetBlock(64, dilation=2)

        self.conv_up1 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn_up1 = nn.BatchNorm2d(128)
        self.block3 = DepthwiseSeparableResNetBlock(128, dilation=1)
        self.block4 = DepthwiseSeparableResNetBlock(128, dilation=2)

        self.conv_up2 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn_up2 = nn.BatchNorm2d(256)
        self.block5 = DepthwiseSeparableResNetBlock(256, dilation=1)
        self.block6 = DepthwiseSeparableResNetBlock(256, dilation=2)

        self.conv_out = nn.Conv2d(256, self.n_states * self.max_bits, kernel_size=3, padding=1)

        self.gnn = TrellisMessagePassing(n_states=self.n_states, hidden_dim=64, max_iterations=6)

    def forward(self, rx: torch.Tensor, snr_map: torch.Tensor, phase: Phase,
                data_mask: torch.Tensor = None, bit_mask: torch.Tensor = None,
                halt_threshold: float = None, max_iterations_override: int = None) -> torch.Tensor:
        batch_size, _, S, n_sc = rx.shape
        rx_with_snr = torch.cat([rx, snr_map], dim=1)

        x = F.relu(self.bn_in(self.conv_in(rx_with_snr)))
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn_up1(self.conv_up1(x)))
        x = self.block3(x)
        x = self.block4(x)
        x = F.relu(self.bn_up2(self.conv_up2(x)))
        x = self.block5(x)
        x = self.block6(x)
        priors_grid = self.conv_out(x)

        priors = priors_grid.view(batch_size, self.max_bits, self.n_states, S, n_sc).permute(0, 1, 3, 4, 2)
        priors_seq = priors.contiguous().view(batch_size * self.max_bits, S * n_sc, self.n_states)

        refined_states_seq = self.gnn(
            priors_seq, data_mask=data_mask, bit_mask=bit_mask,
            halt_threshold=halt_threshold, max_iterations_override=max_iterations_override,
            max_bits=self.max_bits,
        )
        refined_states = refined_states_seq.view(batch_size, self.max_bits, S, n_sc, self.n_states)

        if phase == Phase.TEST:
            detected_states = torch.argmax(refined_states, dim=-1)
            return (detected_states % 2).float()
        return refined_states

    def forward_with_snapshots(self, rx: torch.Tensor, snr_map: torch.Tensor, data_mask: torch.Tensor = None):
        """
        Diagnostic-only wrapper: mirrors forward()'s CNN frontend + reshape logic,
        but calls TrellisMessagePassing.forward_with_snapshots to get per-iteration
        decoded bits instead of only the final iteration's output.
        Returns a list of `max_iterations` tensors, each shaped (batch, max_bits, S, F),
        already thresholded to 0/1 bits (matching Phase.TEST behavior) for direct
        use with compute_ber.
        """
        batch_size, _, S, n_sc = rx.shape

        rx_with_snr = torch.cat([rx, snr_map], dim=1)

        # FIX: was calling self.conv_up/self.bn_up (don't exist on this class)
        # and only running block3/block4 — now matches the real 6-block frontend.
        x = F.relu(self.bn_in(self.conv_in(rx_with_snr)))
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn_up1(self.conv_up1(x)))
        x = self.block3(x)
        x = self.block4(x)
        x = F.relu(self.bn_up2(self.conv_up2(x)))
        x = self.block5(x)
        x = self.block6(x)
        priors_grid = self.conv_out(x)

        priors = priors_grid.view(batch_size, self.max_bits, self.n_states, S, n_sc).permute(0, 1, 3, 4, 2)
        priors_seq = priors.contiguous().view(batch_size * self.max_bits, S * n_sc, self.n_states)

        raw_snapshots = self.gnn.forward_with_snapshots(priors_seq, data_mask=data_mask)

        decoded_snapshots = []
        for iter_output in raw_snapshots:
            refined_states = iter_output.view(batch_size, self.max_bits, S, n_sc, self.n_states)
            detected_states = torch.argmax(refined_states, dim=-1)
            detected_bits = (detected_states % 2).float()
            decoded_snapshots.append(detected_bits)

        return decoded_snapshots