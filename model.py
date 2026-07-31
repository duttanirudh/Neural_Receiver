import torch
import torch.nn as nn


class NeuralReceiver(nn.Module):

    def __init__(
        self,
        hidden_channels=64,
    ):
        super().__init__()


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

        self.classifier = nn.Conv1d(
            hidden_channels,
            2,
            kernel_size=1,
        )

    def forward(self, rx):

        x = torch.stack(
            (
                rx.real,
                rx.imag,
            ),
            dim=1,
        )

        x = self.features(x)

        x = self.classifier(x)

        x = x.permute(0, 2, 1)

        x = x.reshape(
            x.shape[0],
            -1,
        )

        return x