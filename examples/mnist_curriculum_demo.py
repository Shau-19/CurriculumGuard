import sys,os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler

# ----------------------------
# Noisy FashionMNIST
# ----------------------------
class NoisyFashion(Dataset):
    def __init__(self, train=True, noise=0.35):
        base = FashionMNIST("examples\data", train=train, download=True, transform=ToTensor())
        self.x = base.data.float().view(len(base),-1)/255
        self.y = base.targets.clone()

        if train:
            for i in random.sample(range(len(self.y)), int(noise*len(self.y))):
                self.y[i] = random.randint(0,9)

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
model = nn.Sequential(
    nn.Linear(784,256), nn.ReLU(),
    nn.Linear(256,128), nn.ReLU(),
    nn.Linear(128,10)
)

opt = torch.optim.Adam(model.parameters(),1e-3)
crit = nn.CrossEntropyLoss(reduction="none")

train_ds = NoisyFashion(train=True)
val_ds   = NoisyFashion(train=False, noise=0.0)

# ----------------------------
# Baseline
# ----------------------------
print("=== Baseline Vision ===")
for e in range(3):
    for _,x,y in DataLoader(train_ds,128,shuffle=True):
        loss = crit(model(x),y)
        loss.mean().backward();opt.step();opt.zero_grad()
    print("epoch",e,"val acc:",accuracy(model,val_ds))

# ----------------------------
# CurriculumGuard
# ----------------------------
print("\n=== CurriculumGuard Vision ===")
guard = CurriculumGuard(train_ds)

for e in range(8):
    sampler = AdaptiveSampler(train_ds,guard.bucketer.bucketize(),guard.weights)
    for ids,x,y in DataLoader(train_ds,128,sampler=sampler):
        out=model(x);loss=crit(out,y)
        guard.profiler.update(ids,loss,out,y)
        loss.mean().backward();opt.step();opt.zero_grad()
    print("epoch",e,"val acc:",accuracy(model,val_ds))
