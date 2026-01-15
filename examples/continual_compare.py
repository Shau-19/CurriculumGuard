import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader

from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler

# ----------------------------
# Two time-distributions
# ----------------------------
class UsersA(Dataset):
    def __init__(self):
        self.x = torch.randn(3000,10)
        self.y = (self.x.sum(1) > 0).long()
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return i,self.x[i],self.y[i]

class UsersB(Dataset):
    def __init__(self):
        self.x = torch.randn(3000,10) + 2
        self.y = (self.x.sum(1) > 6).long()
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return i,self.x[i],self.y[i]

# ----------------------------
# Accuracy
# ----------------------------
def accuracy(model,ds):
    c=0;t=0
    with torch.no_grad():
        for _,x,y in DataLoader(ds,256):
            p=model(x).argmax(1)
            c+=(p==y).sum().item()
            t+=len(y)
    return c/t

# ----------------------------
# Model
# ----------------------------
model = nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,2))
opt = torch.optim.Adam(model.parameters(),1e-2)
crit = nn.CrossEntropyLoss(reduction="none")

# ----------------------------
# Baseline
# ----------------------------
print("=== Baseline Continual Learning ===")
dsA, dsB = UsersA(), UsersB()

for e in range(4):
    for _,x,y in DataLoader(dsA,64,shuffle=True):
        loss=crit(model(x),y)
        loss.mean().backward();opt.step();opt.zero_grad()
    print("Task-A epoch",e,"acc:",accuracy(model,dsA))

print("After shift to Task-B")
for e in range(6):
    for _,x,y in DataLoader(dsB,64,shuffle=True):
        loss=crit(model(x),y)
        loss.mean().backward();opt.step();opt.zero_grad()
    print("Task-B epoch",e,"acc:",accuracy(model,dsB))

# ----------------------------
# CurriculumGuard
# ----------------------------
print("\n=== CurriculumGuard Continual Learning ===")
model = nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,2))
opt = torch.optim.Adam(model.parameters(),1e-2)
guard = CurriculumGuard(dsA)

for e in range(4):
    sampler=AdaptiveSampler(dsA,guard.bucketer.bucketize(),guard.weights)
    for ids,x,y in DataLoader(dsA,64,sampler=sampler):
        out=model(x);loss=crit(out,y)
        guard.profiler.update(ids,loss,out,y)
        loss.mean().backward();opt.step();opt.zero_grad()
    print("Task-A epoch",e,"acc:",accuracy(model,dsA))

print("After shift to Task-B")

guard.dataset = dsB
for e in range(6):
    sampler=AdaptiveSampler(dsB,guard.bucketer.bucketize(),guard.weights)
    for ids,x,y in DataLoader(dsB,64,sampler=sampler):
        out=model(x);loss=crit(out,y)
        guard.profiler.update(ids,loss,out,y)
        loss.mean().backward();opt.step();opt.zero_grad()
    print("Task-B epoch",e,"acc:",accuracy(model,dsB))
