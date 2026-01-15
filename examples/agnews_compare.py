import sys, os, torch, random, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "examples", "data", "train.csv")
from torch import nn
from torch.utils.data import Dataset, DataLoader
from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "examples", "data", "train.csv")

# ----------- FAST TEXT ENCODER -----------
def fast_encode(t, maxlen=200):
    arr = torch.frombuffer(bytes(t.lower()[:maxlen], "utf-8", "ignore"), dtype=torch.uint8)
    return torch.bincount(arr % 200, minlength=200).float()

# ----------- DATASET -----------
class AGNewsBinary(Dataset):
    def __init__(self, limit=4000, noise=1200):
        print("Loading AG News CSV...")

        self.x, self.y = [], []

        with open(DATA_PATH, encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header row

            for i, row in enumerate(reader):
                if i >= limit:
                    break
                label = int(row[0]) - 1
                text = row[1] + " " + row[2]
                self.x.append(fast_encode(text))
                self.y.append(label % 2)

        for _ in range(noise):
            self.x.append(torch.randn(200))
            self.y.append(random.randint(0,1))

        print("Dataset ready:", len(self.x))

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return i, self.x[i], self.y[i]

# ----------- ACCURACY -----------
def accuracy(model, ds):
    c=t=0
    with torch.no_grad():
        for _, x, y in DataLoader(ds, 128):
            p = model(x).argmax(1)
            c += (p == y).sum().item()
            t += len(y)
    return c / t

# ----------- MAIN -----------
if __name__ == "__main__":
    torch.set_num_threads(2)

    model = nn.Sequential(
        nn.Linear(200,64),
        nn.ReLU(),
        nn.Linear(64,2)
    )

    opt  = torch.optim.Adam(model.parameters(), 1e-3)
    crit = nn.CrossEntropyLoss(reduction="none")

    ds = AGNewsBinary()
    guard = CurriculumGuard(ds)

    print("=== Baseline ===")
    for e in range(3):
        for _, x, y in DataLoader(ds, 128, shuffle=True, num_workers=2, persistent_workers=True):
            loss = crit(model(x), y)
            loss.mean().backward()
            opt.step(); opt.zero_grad()
        print("epoch", e, "acc:", accuracy(model, ds))

    print("\n=== CurriculumGuard ===")
    for e in range(8):
        sampler = AdaptiveSampler(ds, guard.bucketer.bucketize(), guard.weights)
        for ids, x, y in DataLoader(ds, 128, sampler=sampler, num_workers=2, persistent_workers=True):
            out  = model(x)
            loss = crit(out, y)

            # critical: prevent autograd graph leak
            guard.profiler.update(ids, loss.detach(), out.detach(), y)

            loss.mean().backward()
            opt.step(); opt.zero_grad()

        print("epoch", e, "acc:", accuracy(model, ds))
