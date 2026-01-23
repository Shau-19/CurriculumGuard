import sys,os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from curriculum_guard import Curriculum


# ----------------------------
# Noisy FashionMNIST Dataset
# ----------------------------
class NoisyFashion(Dataset):
    def __init__(self, train=True, noise=0.35):
        base = FashionMNIST(
            "examples/data",
            train=train,
            download=True,
            transform=ToTensor()
        )

        self.x = base.data.float().view(len(base), -1) / 255.0
        self.y = base.targets.clone()

        # inject label noise
        if train:
            for i in random.sample(range(len(self.y)), int(noise * len(self.y))):
                self.y[i] = random.randint(0, 9)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


# ----------------------------
# Accuracy
# ----------------------------
def accuracy(model, ds):
    correct = total = 0
    with torch.no_grad():
        for _, x, y in DataLoader(ds, 256):
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


# ----------------------------
# Model
# ----------------------------
def make_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


criterion = nn.CrossEntropyLoss(reduction="none")


# ----------------------------
# Data
# ----------------------------
train_ds = NoisyFashion(train=True, noise=0.35)
val_ds   = NoisyFashion(train=False, noise=0.0)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)


# ============================
# 1️⃣ Baseline Training
# ============================
print("=== Baseline Vision ===")

model = make_model()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

for epoch in range(3):
    for _, x, y in train_loader:
        loss = criterion(model(x), y)
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {epoch} | val acc: {accuracy(model, val_ds):.4f}")


# ============================
# 2️⃣ CurriculumGuard v0.2
# ============================
print("\n=== CurriculumGuard Vision ===")

model = make_model()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

curriculum = Curriculum.auto(train_ds)

for epoch in range(8):
    for ids, x, y in curriculum(train_loader):
        logits = model(x)
        loss = criterion(logits, y)

        # single curriculum hook
        curriculum.step(ids, loss, logits, y)

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {epoch} | val acc: {accuracy(model, val_ds):.4f}")
