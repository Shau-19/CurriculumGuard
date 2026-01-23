import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from curriculum_guard import Curriculum


# ----------------------------
# Two time distributions
# ----------------------------
class UsersA(Dataset):
    def __init__(self):
        self.x = torch.randn(3000, 10)
        self.y = (self.x.sum(1) > 0).long()

    def __len__(self): 
        return len(self.x)

    def __getitem__(self, i): 
        return i, self.x[i], self.y[i]


class UsersB(Dataset):
    def __init__(self):
        self.x = torch.randn(3000, 10) + 2
        self.y = (self.x.sum(1) > 6).long()

    def __len__(self): 
        return len(self.x)

    def __getitem__(self, i): 
        return i, self.x[i], self.y[i]


# ----------------------------
# Accuracy
# ----------------------------
def accuracy(model, ds):
    c = t = 0
    with torch.no_grad():
        for _, x, y in DataLoader(ds, 256):
            p = model(x).argmax(1)
            c += (p == y).sum().item()
            t += len(y)
    return c / t


# ----------------------------
# Model
# ----------------------------
def make_model():
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )


criterion = nn.CrossEntropyLoss(reduction="none")


# ===========================
# 1️⃣ Baseline Continual Learning
# ===========================
print("=== Baseline Continual Learning ===")

model = make_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

dsA, dsB = UsersA(), UsersB()

for e in range(4):
    for _, x, y in DataLoader(dsA, 64, shuffle=True):
        loss = criterion(model(x), y)
        loss.mean().backward()
        opt.step(); opt.zero_grad()
    print(f"Task-A epoch {e} acc:", accuracy(model, dsA))

print("After shift to Task-B")

for e in range(6):
    for _, x, y in DataLoader(dsB, 64, shuffle=True):
        loss = criterion(model(x), y)
        loss.mean().backward()
        opt.step(); opt.zero_grad()
    print(f"Task-B epoch {e} acc:", accuracy(model, dsB))


# ===========================
# 2️⃣ CurriculumGuard v0.2
# ===========================
print("\n=== CurriculumGuard Continual Learning ===")

model = make_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# Phase A
curriculum = Curriculum.auto(dsA)
loaderA = DataLoader(dsA, 64, shuffle=True)

for e in range(4):
    for ids, x, y in curriculum(loaderA):
        logits = model(x)
        loss = criterion(logits, y)
        curriculum.step(ids, loss, logits, y)

        loss.mean().backward()
        opt.step(); opt.zero_grad()

    print(f"Task-A epoch {e} acc:", accuracy(model, dsA))

# Phase B (distribution shift)
print("After shift to Task-B")

curriculum = Curriculum.auto(dsB)
loaderB = DataLoader(dsB, 64, shuffle=True)

for e in range(6):
    for ids, x, y in curriculum(loaderB):
        logits = model(x)
        loss = criterion(logits, y)
        curriculum.step(ids, loss, logits, y)

        loss.mean().backward()
        opt.step(); opt.zero_grad()

    print(f"Task-B epoch {e} acc:", accuracy(model, dsB))
