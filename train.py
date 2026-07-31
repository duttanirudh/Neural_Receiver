import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset
from model import NeuralReceiver



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


BATCH_SIZE = 2048

BITS_PER_MESSAGE = 256

LEARNING_RATE = 1e-3

EPOCHS = 50

MODULATION = "QPSK"



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


model = NeuralReceiver().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

def train():
    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        batches = 0

        for rx, bits in loader:

            prediction = model(rx)

            loss = criterion(
                prediction,
                bits.float(),
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            running_loss += loss.item()

            batches += 1


            if batches == 500:
                break


        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss = {running_loss/batches:.6f}"
        )


        torch.save(
            model.state_dict(),
            "receiver.pt",
        )

    print("Training Complete.")