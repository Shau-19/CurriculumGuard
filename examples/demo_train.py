import sys, os, torch, random
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from torch import nn
from torch.utils.data import Dataset, DataLoader
 
from curriculum_guard import Curriculum
 
 
# ---------------------------
# Noisy Toy Dataset
# ---------------------------
class NoisyDataset(Dataset):
    def __init__(self, n=4000, noise_rate=0.3):
        self.x = torch.randn(n, 10)
        self.y = (self.x.sum(dim=1) > 0).long()
 
        # inject label noise
        if noise_rate > 0:
            for _ in range(int(noise_rate * n)):
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
        for batch in loader:
            # Handle both (id, x, y) and (x, y) formats
            if len(batch) == 3:
                _, x, y = batch
            else:
                x, y = batch
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
 
 
# ---------------------------
# Setup
# ---------------------------
train_ds = NoisyDataset(n=4000, noise_rate=0.3)  # 30% noise
val_ds = NoisyDataset(n=1000, noise_rate=0.0)    # Clean validation!
 
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
 
criterion = nn.CrossEntropyLoss(reduction="none")
 
 
# ===========================
# 1. Baseline Training
# ===========================
print("\n=== Baseline Training ===")
model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
for epoch in range(5):
    model.train()
    total_loss = 0.0
 
    for _, x, y in train_loader:
        out = model(x)
        loss = criterion(out, y)
 
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
 
        total_loss += loss.mean().item()
 
    model.eval()
    train_acc = accuracy(model, train_loader)
    val_acc = accuracy(model, val_loader)  # Test on CLEAN data!
    print(f"Epoch {epoch:02d} | Loss: {total_loss:.2f} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")
 
 
# ===========================
# 2. CurriculumGuard Training (v0.2)
# ===========================
print("\n=== CurriculumGuard Training ===")
 
model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
curriculum = Curriculum.auto(train_ds)
 
for epoch in range(10):
    model.train()
    total_loss = 0.0
 
    for ids, x, y in curriculum(train_loader):
        out = model(x)
        loss = criterion(out, y)
 
        # curriculum feedback
        curriculum.step(ids, loss, out, y)
 
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
 
        total_loss += loss.mean().item()
 
    # Validation loop for step_validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _, x, y in val_loader:
            out = model(x)
            loss = criterion(out, y)
            val_losses.extend(loss.tolist())
    val_loss = sum(val_losses) / len(val_losses)
    
    # safety / policy validation feedback
    curriculum.step_validation(val_loss)
 
    train_acc = accuracy(model, train_loader)
    val_acc = accuracy(model, val_loader)  # Test on CLEAN data!
    stats = curriculum.stats()
    print(f"Epoch {epoch:02d} | Loss: {total_loss:.2f} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | Weights: {stats['weights']}")
 
print("\n" + "="*60)
print("KEY INSIGHT:")
print("- Train Acc = performance on NOISY labels (misleading!)")
print("- Val Acc   = performance on CLEAN labels (real metric!)")
print("="*60)
