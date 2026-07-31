import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset
from model import NeuralReceiver


############################################################
# Device
############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


############################################################
# Hyperparameters
############################################################

BATCH_SIZE = 2048

BITS_PER_MESSAGE = 256

LEARNING_RATE = 1e-3

EPOCHS = 50

MODULATION = "QPSK"


############################################################
# Dataset
############################################################

dataset = Dataset(
    batch_size=BATCH_SIZE,
    bits_per_message=BITS_PER_MESSAGE,
    modulation=MODULATION,
    device=device,
)

loader = DataLoader(
    dataset,
    batch_size=None,
)


############################################################
# Model
############################################################

model = NeuralReceiver().to(device)


############################################################
# Loss
############################################################

criterion = nn.BCEWithLogitsLoss()


############################################################
# Optimizer
############################################################

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


############################################################
# Training Loop
############################################################
def train():
    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        batches = 0

        ########################################################
        # Since IterableDataset is infinite,
        # manually decide how many batches make one epoch.
        ########################################################

        for rx, bits in loader:

            ####################################################
            # Forward
            ####################################################

            prediction = model(rx)

            loss = criterion(
                prediction,
                bits.float(),
            )

            ####################################################
            # Backprop
            ####################################################

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            ####################################################
            # Statistics
            ####################################################

            running_loss += loss.item()

            batches += 1

            ####################################################
            # Define Epoch Length
            ####################################################

            if batches == 500:
                break

        ########################################################
        # Print Progress
        ########################################################

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss = {running_loss/batches:.6f}"
        )

        ########################################################
        # Save Checkpoint
        ########################################################

        torch.save(
            model.state_dict(),
            "receiver.pt",
        )

    print("Training Complete.")