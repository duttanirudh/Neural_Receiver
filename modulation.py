import torch
import math


def modulate(bits):

    return(qpsk_modulate(bits))


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
