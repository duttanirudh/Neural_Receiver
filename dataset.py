import torch
from torch.utils.data import IterableDataset
from modulation import modulate
from channels import apply_channel


class Dataset(IterableDataset):
    
    #generate random bits, modulate them, pass through channel, return received signal and original bits

    def __init__(
        self,
        batch_size=1024,
        bits_per_message=256,
        modulation="QPSK",
        device="cuda",
        return_channel=False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.bits_per_message = bits_per_message
        self.modulation = modulation
        self.device = device
        self.return_channel = return_channel

    def generate_bits(self):
        
        return torch.randint(
            low=0,
            high=2,
            size=(self.batch_size, self.bits_per_message),
            device=self.device,
            dtype=torch.int8,
        )

    def __iter__(self):

        while True:

        
            bits = self.generate_bits()


            tx_symbols = modulate(
                bits,
                modulation=self.modulation,
            )

            rx_symbols, channel_info = apply_channel(
                tx_symbols
            )

            if self.return_channel:

                yield {
                    "received": rx_symbols,
                    "bits": bits,
                    "transmitted": tx_symbols,
                    "channel": channel_info,
                }

            else:

                yield rx_symbols, bits