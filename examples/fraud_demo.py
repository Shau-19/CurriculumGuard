import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from curriculum_guard import Curriculum


# ---------------------------
# Fraud Dataset
# ---------------------------
class FraudDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("examples/data/creditcard.csv")

        # use first 50k rows for speed
        self.x = torch.tensor(
            df.iloc[:50000, 1:30].values,
            dtype=torch.float32
        )
        self.y = torch.tensor(
            df.iloc[:50000, -1].values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


# ---------------------------
# Fraud Recall (minority class)
# ---------------------------
def fraud_recall(model, ds):
    tp, fn = 0, 0
    with torch.no_grad():
        for _, x, y in DataLoader(ds, 256):
            pred = model(x).argmax(1)
            for p, t in zip(pred, y):
                if t == 1 and p == 1:
                    tp += 1
                elif t == 1 and p == 0:
                    fn += 1
    return tp / (tp + fn + 1e-8)


# ---------------------------
# Model
# ---------------------------
def make_model():
    return nn.Sequential(
        nn.Linear(29, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )


criterion = nn.CrossEntropyLoss(reduction="none")


# ---------------------------
# Data
# ---------------------------
ds = FraudDataset()
loader = DataLoader(ds, batch_size=128, shuffle=True)


# ===========================
# 1️⃣ Baseline Training
# ===========================
print("=== Baseline Fraud ===")

model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    for _, x, y in loader:
        loss = criterion(model(x), y)
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {epoch} | fraud recall: {fraud_recall(model, ds):.4f}")


# ===========================
# 2️⃣ CurriculumGuard v0.2
# ===========================
print("\n=== CurriculumGuard Fraud ===")

model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

curriculum = Curriculum.auto(ds)

for epoch in range(8):
    for ids, x, y in curriculum(loader):
        logits = model(x)
        loss = criterion(logits, y)

        # single curriculum hook
        curriculum.step(ids, loss, logits, y)

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {epoch} | fraud recall: {fraud_recall(model, ds):.4f}")
