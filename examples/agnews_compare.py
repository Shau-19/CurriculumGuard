import sys, os, torch, random, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import os, torch, random, csv
from torch import nn
from torch.utils.data import Dataset, DataLoader

from curriculum_guard import Curriculum


# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "examples", "data", "train.csv")


# ---------------------------
# FAST TEXT ENCODER
# ---------------------------
def fast_encode(t, maxlen=200):
    arr = torch.frombuffer(
        bytes(t.lower()[:maxlen], "utf-8", "ignore"),
        dtype=torch.uint8
    )
    return torch.bincount(arr % 200, minlength=200).float()


# ---------------------------
# DATASET
# ---------------------------
class AGNewsBinary(Dataset):
    def __init__(self, limit=4000, noise=1200):
        print("Loading AG News CSV...")

        self.x, self.y = [], []

        with open(DATA_PATH, encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)

            for i, row in enumerate(reader):
                if i >= limit:
                    break
                label = int(row[0]) - 1
                text = row[1] + " " + row[2]
                self.x.append(fast_encode(text))
                self.y.append(label % 2)

        # inject noise
        for _ in range(noise):
            self.x.append(torch.randn(200))
            self.y.append(random.randint(0, 1))

        print("Dataset ready:", len(self.x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


# ---------------------------
# ACCURACY
# ---------------------------
def accuracy(model, ds):
    correct = total = 0
    with torch.no_grad():
        for _, x, y in DataLoader(ds, 128):
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    torch.set_num_threads(2)

    model = nn.Sequential(
        nn.Linear(200, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")

    ds = AGNewsBinary()
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    # ---------------------------
    # BASELINE
    # ---------------------------
    print("\n=== Baseline ===")
    for epoch in range(3):
        for _, x, y in loader:
            loss = criterion(model(x), y)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch} acc:", accuracy(model, ds))

    # ---------------------------
    # CURRICULUMGUARD v0.2
    # ---------------------------
    print("\n=== CurriculumGuard ===")

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

        print(f"epoch {epoch} acc:", accuracy(model, ds))
