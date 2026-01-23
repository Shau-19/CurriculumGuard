import sys, os, torch, random, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader

from curriculum_guard import Curriculum   # v0.2 API


# ---------------------------
# Noisy Toy Dataset
# ---------------------------
class NoisyDataset(Dataset):
    def __init__(self, n=4000):
        self.x = torch.randn(n, 10)
        self.y = (self.x.sum(dim=1) > 0).long()

        # inject label noise (30%)
        for _ in range(int(0.3 * n)):
            idx = random.randint(0, n - 1)
            self.y[idx] = 1 - self.y[idx]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


# ---------------------------
# Model
# ---------------------------
def make_model():
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )


def accuracy(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for _, x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ---------------------------
# Setup
# ---------------------------
train_ds = NoisyDataset()
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss(reduction="none")


# ===========================
# 1️⃣ Baseline Training
# ===========================
print("\n=== Baseline Training ===")
model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0.0

    for _, x, y in train_loader:
        out = model(x)
        loss = criterion(out, y)

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.mean().item()

    acc = accuracy(model, train_loader)
    print(f"Epoch {epoch:02d} | Loss: {total_loss:.2f} | Acc: {acc:.3f}")


# ===========================
# 2️⃣ CurriculumGuard Training (v0.2)
# ===========================
print("\n=== CurriculumGuard Training ===")

model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

curriculum = Curriculum.auto(train_ds)

for epoch in range(10):
    total_loss = 0.0

    for ids, x, y in curriculum(train_loader):
        out = model(x)
        loss = criterion(out, y)

        # 🔑 curriculum feedback
        curriculum.step(ids, loss, out, y)

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.mean().item()

    acc = accuracy(model, train_loader)
    print(f"Epoch {epoch:02d} | Loss: {total_loss:.2f} | Acc: {acc:.3f}")
