import math
import torch
import torch.nn.functional as F

def apply_channel(
        
    tx_symbols,
    snr_db_range=(0, 20),
    attenuation_range=(0.3, 1.0),
    phase_range=(-math.pi, math.pi),
    num_taps=4,
    estimation_error_std=0.05,
):

    rx, channel = apply_frequency_selective_rayleigh(
        tx_symbols,
        num_taps=num_taps,
    )


    rx, attenuation = apply_amplitude_attenuation(
        rx,
        attenuation_range,
    )


    rx, phase = apply_phase_distortion(
        rx,
        phase_range,
    )


    snr_db = torch.empty(
        1,
        device=tx_symbols.device
    ).uniform_(
        snr_db_range[0],
        snr_db_range[1],
    ).item()

    rx = apply_awgn(
        rx,
        snr_db,
    )


    estimated_channel = apply_channel_estimation_error(
        channel,
        estimation_error_std,
    )

    info = {
        "true_channel": channel,
        "estimated_channel": estimated_channel,
        "attenuation": attenuation,
        "phase": phase,
        "snr_db": snr_db,
    }

    return rx, info

def apply_awgn(signal, snr_db):

    power = torch.mean(torch.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = power / snr_linear

    sigma = torch.sqrt(noise_power / 2)

    noise = sigma * (
        torch.randn_like(signal)
        +
        1j * torch.randn_like(signal)
    )

    return signal + noise

def apply_amplitude_attenuation(
    signal,
    attenuation_range,
):

    batch = signal.shape[0]

    alpha = torch.empty(
        batch,
        1,
        device=signal.device,
    ).uniform_(
        attenuation_range[0],
        attenuation_range[1],
    )

    return signal * alpha, alpha

def apply_phase_distortion(
    signal,
    phase_range,
):

    batch = signal.shape[0]

    theta = torch.empty(
        batch,
        1,
        device=signal.device,
    ).uniform_(
        phase_range[0],
        phase_range[1],
    )

    signal = signal * torch.exp(1j * theta)

    return signal, theta


def apply_frequency_selective_rayleigh(
    signal,
    num_taps=4,
):

    batch, length = signal.shape

    real = torch.randn(
        batch,
        num_taps,
        device=signal.device,
    )

    imag = torch.randn(
        batch,
        num_taps,
        device=signal.device,
    )

    taps = torch.complex(real, imag)

    taps = taps / math.sqrt(2 * num_taps)

    output = torch.zeros_like(signal)

    for k in range(num_taps):

        delayed = torch.roll(signal, shifts=k, dims=1)

        delayed[:, :k] = 0

        output += taps[:, k:k+1] * delayed

    return output, taps

def apply_channel_estimation_error(
    channel,
    std,
):

    noise = std * (
        torch.randn_like(channel)
        +
        1j * torch.randn_like(channel)
    )

    return channel + noise