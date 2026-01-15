import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import pandas as pd, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler
class FraudDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("examples/data/creditcard.csv")
        self.x = torch.tensor(df.iloc[:50000,1:30].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:50000,-1].values, dtype=torch.long)

    def __len__(self): return len(self.x)
    def __getitem__(self,i): return i,self.x[i],self.y[i]

def fraud_recall(model, ds):
    loader = DataLoader(ds,256)
    tp, fn = 0, 0
    with torch.no_grad():
        for _,x,y in loader:
            pred = model(x).argmax(1)
            for p,t in zip(pred,y):
                if t==1 and p==1: tp+=1
                if t==1 and p==0: fn+=1
    return tp / (tp+fn+1e-8)

model = nn.Sequential(nn.Linear(29,64),nn.ReLU(),nn.Linear(64,2))
opt = torch.optim.Adam(model.parameters(),1e-3)
crit = nn.CrossEntropyLoss(reduction="none")

ds = FraudDataset()
guard = CurriculumGuard(ds)

print("=== Baseline ===")
for e in range(3):
    for ids,x,y in DataLoader(ds,128,shuffle=True):
        out=model(x); loss=crit(out,y)
        loss.mean().backward(); opt.step(); opt.zero_grad()
    print("epoch",e,"fraud recall:",fraud_recall(model,ds))

print("\n=== CurriculumGuard ===")
for e in range(8):
    sampler=AdaptiveSampler(ds,guard.bucketer.bucketize(),guard.weights)
    for ids,x,y in DataLoader(ds,128,sampler=sampler):
        out=model(x); loss=crit(out,y)
        guard.profiler.update(ids,loss,out,y)
        loss.mean().backward(); opt.step(); opt.zero_grad()
    print("epoch",e,"fraud recall:",fraud_recall(model,ds))
