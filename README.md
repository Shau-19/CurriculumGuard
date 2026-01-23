# 🛡 CurriculumGuard  
**Training-Time Data Control for PyTorch**

[![PyPI](https://img.shields.io/pypi/v/curriculumguard.svg)](https://pypi.org/project/curriculumguard/)  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

CurriculumGuard is an open-source **training-time data control system** for PyTorch that dynamically adapts **which samples a model sees during training** using live learning dynamics — while enforcing stability via rollback-based safety guards.

> Models and optimizers are controlled.  
> Hyperparameters are tuned.  
> **But the data stream itself has been ignored — until now.**

---

## 🔥 Why CurriculumGuard?

Modern datasets are increasingly:
- Noisy  
- Imbalanced  
- Web-scraped  
- Non-stationary  

Yet most training pipelines assume the dataset is **static and trustworthy**.

CurriculumGuard introduces a missing layer in ML infrastructure:

> **Adaptive Data Curriculum with Stability-First Control**

Instead of changing *how* models learn, CurriculumGuard changes **what they learn from — safely, during training**.

---

## ⚙ Installation

```bash
pip install curriculumguard
```

---

## 🚀 Quick Start (v0.2 API)

### 1️⃣ Dataset must return sample IDs

CurriculumGuard needs sample-level identity to track learning dynamics.

```python
def __getitem__(self, idx):
    return idx, data, label
```

---

### 2️⃣ Minimal usage (Beginner)

```python
from curriculum_guard.curriculum import Curriculum

curriculum = Curriculum.auto(train_dataset)

for ids, x, y in curriculum(train_loader):
    logits = model(x)
    loss   = criterion(logits, y)

    curriculum.step(ids, loss, logits, y)

    loss.mean().backward()
    optimizer.step()
    optimizer.zero_grad()
```

That's it.

* No custom samplers
* No weighting logic
* No curriculum math
* Same PyTorch training loop

---

## 🧠 Mental Model

CurriculumGuard acts like an **optimizer for data**:

```
Data → Model → Loss → Curriculum → Safer Data → Model
```

It continuously answers:

> "Which samples are helping learning right now — and which are destabilizing it?"

---

## 🧠 Signals Observed (Automatically)

| Signal             | What It Represents         |
| ------------------ | -------------------------- |
| EMA loss           | Sample difficulty          |
| Loss variance      | Label noise                |
| Prediction entropy | Shortcut learning          |
| Forgetting events  | Unstable / harmful samples |
| Exposure count     | Over-training risk         |

These signals are **observed, not enforced** — safety decisions are made separately.

---

## 🛡 Safety Model

CurriculumGuard is **conservative by design**.

* Curriculum decisions are **advisory**
* Safety mechanisms are **authoritative**
* Harmful curriculum updates are **rolled back**
* Training stability is never sacrificed

> Policy proposes. Safety decides.

---

## 📊 Benchmarks

| Task                     | Baseline        | CurriculumGuard       |
| ------------------------ | --------------- | --------------------- |
| AG News (noisy labels)   | 68%             | **74%**               |
| FashionMNIST (35% noise) | 84%             | **87.5%**             |
| Fraud Detection (recall) | slow & unstable | **fast, high recall** |
| Continual Drift          | fragile         | **stable**            |

---

## 🧩 Progressive API Design (v0.2)

CurriculumGuard scales with user expertise.

### 🟢 Beginner (default)

```python
curriculum = Curriculum.auto(dataset)
```

Safe defaults, minimal setup.

---

### 🟡 Intermediate (optional tuning)

```python
curriculum = Curriculum.auto(
    dataset,
    sensitivity="medium",   # low | medium | high
    warmup_epochs=2,
    safety=True
)
```

---

### 🔵 Advanced (explicit strategies)

```python
curriculum = Curriculum.custom(
    dataset,
    policy="anti_noise",
    bucketing="quantile",
    safety="rollback",
    entropy_weight=0.3
)
```

---

### 🔴 Research-level (full control)

```python
curriculum = Curriculum.from_components(
    profiler=CustomProfiler(),
    policy=MyPolicy(),
    safety=MySafetyController(),
    bucketer=MyBucketer()
)
```

---

## 🧪 Where CurriculumGuard Shines

* Noisy labels
* Long training runs
* Expensive experiments
* Continual / non-stationary data
* High-risk domains (fraud, medical, finance)

If your dataset is clean, CurriculumGuard stays out of the way.

If it's not — it stabilizes learning.

---

## 📜 License

MIT