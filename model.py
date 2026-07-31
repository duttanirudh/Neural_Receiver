import torch
import torch.nn as nn


class NeuralReceiver(nn.Module):

    def __init__(
        self,
        hidden_channels=64,
    ):
        super().__init__()

        ####################################################
        # Feature Extractor
        ####################################################

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=2,
                out_channels=hidden_channels,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),
        )

        ####################################################
        # Bit Prediction
        ####################################################

        self.classifier = nn.Conv1d(
            hidden_channels,
            2,
            kernel_size=1,
        )

    def forward(self, rx):

        ####################################################
        # Complex → 2 Channels
        ####################################################

        x = torch.stack(
            (
                rx.real,
                rx.imag,
            ),
            dim=1,
        )

        ####################################################
        # CNN
        ####################################################

        x = self.features(x)

        ####################################################
        # Predict Two Bits Per Symbol (QPSK)
        ####################################################

        x = self.classifier(x)

        ####################################################
        # Shape
        #
        # (batch,2,num_symbols)
        # ↓
        # (batch,num_symbols,2)
        ####################################################

        x = x.permute(0, 2, 1)

        ####################################################
        # Flatten
        ####################################################

        x = x.reshape(
            x.shape[0],
            -1,
        )

        return x