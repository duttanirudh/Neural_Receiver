import torch

from dataset import Dataset
from model import NeuralReceiver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Dataset(
    batch_size=16,
    bits_per_message=256,
    modulation="QPSK",
    device=device,
)

rx, bits = next(iter(dataset))

print("Received Signal")
print(rx.shape)
print(rx.dtype)

print()

print("Ground Truth Bits")
print(bits.shape)
print(bits.dtype)

model = NeuralReceiver().to(device)

output = model(rx)

print()

print("Model Output")
print(output.shape)

assert output.shape == bits.shape

loss = torch.nn.BCEWithLogitsLoss()

L = loss(output, bits.float())

print()

print("Loss =", L.item())

L.backward()

print("Backward pass successful.")