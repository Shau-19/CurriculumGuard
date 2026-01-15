import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler
# ---------------------------
# Noisy Toy Dataset
# ---------------------------
class NoisyDataset(Dataset):
    def __init__(self,n=4000):
        self.x=torch.randn(n,10)
        self.y=(self.x.sum(dim=1)>0).long()

        # inject noise
        for i in range(int(0.3*n)):
            idx=random.randint(0,n-1)
            self.y[idx]=1-self.y[idx]

    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        return i,self.x[i],self.y[i]

# ---------------------------
# Model
# ---------------------------
model=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,2))
opt=torch.optim.Adam(model.parameters(),lr=0.01)
crit=nn.CrossEntropyLoss(reduction="none")

train_ds=NoisyDataset()
val_ds=NoisyDataset(1000)

# ---------------------------
# 1️⃣ Baseline Training
# ---------------------------
print("\n=== Baseline Training ===")
for epoch in range(5):
    loader=DataLoader(train_ds,batch_size=64,shuffle=True)
    tot=0
    for ids,x,y in loader:
        out=model(x)
        loss=crit(out,y)
        loss.mean().backward()
        opt.step();opt.zero_grad()
        tot+=loss.mean().item()
    print(f"Epoch {epoch} loss:",tot)

# ---------------------------
# 2️⃣ CurriculumGuard Training
# ---------------------------
print("\n=== CurriculumGuard Training ===")
guard=CurriculumGuard(train_ds)

for epoch in range(10):
    buckets=guard.bucketer.bucketize()
    sampler=AdaptiveSampler(train_ds,buckets,guard.weights)
    loader=DataLoader(train_ds,batch_size=64,sampler=sampler)

    tot=0
    for ids,x,y in loader:
        out=model(x)
        loss=crit(out,y)
        guard.profiler.update(ids,loss,out,y)

        loss.mean().backward()
        opt.step();opt.zero_grad()
        tot+=loss.mean().item()

    print(f"Epoch {epoch} loss:",tot)
