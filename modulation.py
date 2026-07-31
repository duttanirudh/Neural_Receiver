import torch
import math


def modulate(bits, modulation="QPSK"):
    """
    Modulate binary bits into complex symbols.

    Parameters
    ----------
    bits : torch.Tensor
        Shape: (batch_size, bits_per_message)
        dtype: int8 or int64

    modulation : str
        "BPSK"
        "QPSK"

    Returns
    -------
    symbols : torch.Tensor
        Complex tensor
    """

    modulation = modulation.upper()

    if modulation == "BPSK":
        return bpsk_modulate(bits)

    elif modulation == "QPSK":
        return qpsk_modulate(bits)

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")


##########################################################
# BPSK
##########################################################

def bpsk_modulate(bits):
    """
    0 -> -1
    1 -> +1
    """

    symbols = 2 * bits.float() - 1

    return symbols.to(torch.complex64)


##########################################################
# QPSK
##########################################################

def qpsk_modulate(bits):
    """
    Gray Coding

    00 -> +1 + j
    01 -> -1 + j
    11 -> -1 - j
    10 -> +1 - j
    """

    if bits.shape[1] % 2 != 0:
        raise ValueError("QPSK requires an even number of bits.")

    b0 = bits[:, 0::2].float()
    b1 = bits[:, 1::2].float()

    # Gray mapping
    real = 1 - 2 * b1
    imag = 1 - 2 * b0

    symbols = torch.complex(real, imag)

    # Normalize average symbol energy to 1
    symbols = symbols / math.sqrt(2)

    return symbols.to(torch.complex64)


##########################################################
# Optional Demodulators
##########################################################

def demodulate(symbols, modulation="QPSK"):

    modulation = modulation.upper()

    if modulation == "BPSK":
        return bpsk_demodulate(symbols)

    elif modulation == "QPSK":
        return qpsk_demodulate(symbols)

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")


def bpsk_demodulate(symbols):

    bits = (symbols.real > 0).to(torch.int8)

    return bits


def qpsk_demodulate(symbols):

    real = (symbols.real < 0).to(torch.int8)
    imag = (symbols.imag < 0).to(torch.int8)

    # Reverse Gray mapping
    b0 = imag
    b1 = real

    bits = torch.empty(
        symbols.shape[0],
        symbols.shape[1] * 2,
        device=symbols.device,
        dtype=torch.int8,
    )

    bits[:, 0::2] = b0
    bits[:, 1::2] = b1

    return bits