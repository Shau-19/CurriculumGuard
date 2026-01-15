# 🛡 CurriculumGuard  
**Training-Time Data Control for PyTorch**

[![PyPI](https://img.shields.io/pypi/v/curriculumguard.svg)](https://pypi.org/project/curriculumguard/)  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

CurriculumGuard is an open-source training-time control system that dynamically adapts **which samples a model sees during training** using live learning dynamics — while enforcing stability via rollback-based safety guards.

> Models and optimizers are controlled.  
> Hyperparameters are tuned.  
> **But the data stream itself has been ignored — until now.**

---

## 🔥 Why CurriculumGuard?

Modern datasets are:
- Noisy  
- Imbalanced  
- Web-scraped  
- Non-stationary  

CurriculumGuard introduces a missing layer in ML infrastructure:

> **Adaptive Data Curriculum with Stability-First Control**

---

## ⚙ Installation

```bash
pip install curriculumguard
```

---

## 🚀 Quick Start

### Dataset must return sample IDs

```python
def __getitem__(self, idx):
    return idx, data, label
```

### Wrap training loop

```python
from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler

guard = CurriculumGuard(train_dataset)

for epoch in range(epochs):
    sampler = AdaptiveSampler(train_dataset, guard.bucketer.bucketize(), guard.weights)
    loader = DataLoader(train_dataset, sampler=sampler)

    for ids, x, y in loader:
        logits = model(x)
        loss   = criterion(logits, y)
        guard.profiler.update(ids, loss.detach(), logits.detach(), y)
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 🧠 Signals Observed

| Signal | What It Represents |
|-------|--------------------|
| EMA loss | Sample difficulty |
| Loss variance | Label noise |
| Prediction entropy | Shortcut learning |
| Forgetting events | Harmful samples |
| Exposure count | Over-training risk |

---

## 🛡 Safety Model

CurriculumGuard enforces rollback when instability or regression is detected.

> Policy is advisory. Safety is authoritative.

---

## 📊 Benchmarks

| Task | Baseline | CurriculumGuard |
|------|----------|----------------|
| AG News + Noise | 68% | **74%** |
| FashionMNIST 35% noise | 84% | **87.5%** |
| Fraud Recall | slow | **fast high recall** |
| Continual Drift | fragile | **stable** 
|
---

## 📊 Real-World Performance

CurriculumGuard was evaluated across four real-world failure modes: noisy labels, garbage web text, class imbalance, and continual distribution shift.

---

### 🧪 1️⃣ NLP — AG News with Garbage Web Text

| Epoch | Baseline Accuracy | CurriculumGuard Accuracy |
|------:|------------------|--------------------------|
| 0 | 0.64 | 0.59 |
| 2 | 0.69 | 0.70 |
| 5 | — | **0.72** |
| 7 | — | **0.739** |

**Observation**

Baseline training plateaus early due to noisy web text.
CurriculumGuard keeps improving by suppressing unstable samples.

---

### 🧪 2️⃣ Vision — FashionMNIST with 35% Label Noise

| Epoch | Baseline Accuracy | CurriculumGuard Accuracy |
|------:|------------------|--------------------------|
| 0 | 0.837 | **0.850** |
| 2 | 0.840 | **0.859** |
| 7 | — | **0.875** |

**Observation**

Label noise stalls conventional training.
CurriculumGuard dynamically downweights corrupted samples.

---

### 🧪 3️⃣ Fraud Detection — Credit Card Transactions

| Epoch | Baseline Recall | CurriculumGuard Recall |
|------:|-----------------|------------------------|
| 0 | 0.44 | **0.66** |
| 2 | 0.86 | **0.88** |
| 5 | — | **0.90** |

**Observation**

CurriculumGuard rapidly improves minority-class recall
without destabilizing training.

---

### 🧪 4️⃣ Continual Learning — Distribution Shift

| Phase | Baseline Accuracy | CurriculumGuard Accuracy |
|-------|------------------|--------------------------|
| Task-A | 0.99 | 0.98 |
| Task-B | 1.00 | **1.00 (no regression)** |

**Observation**

Both systems adapt quickly, but CurriculumGuard enforces
safety guarantees under distribution drift.

---

### 📥 Dataset Handling

Fraud_Demo(ANN): Credit Card Fraud Detection Dataset(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

AGNews(NLP): AG News Classification Dataset(https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data?select=train.csv)

MNIST(Vision ANN): FashionMNIST Dataset from sklearn.datasets(Downloads as you run the script)


---
## 📜 License

MIT

