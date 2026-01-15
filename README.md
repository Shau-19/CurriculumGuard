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
| Continual Drift | fragile | **stable** |

---

## 📜 License

MIT
