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

It works entirely inside your training loop — no restarts, no trial explosion.

---

## ⚙ Installation

```bash
pip install curriculumguard
```

Verify installation:

```bash
python - <<EOF
from curriculum_guard.core.guard import CurriculumGuard
print(CurriculumGuard)
EOF
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

## 📊 Real-World Performance

CurriculumGuard was evaluated across four real-world failure modes: noisy labels, garbage web text, class imbalance, and continual distribution shift.

---

### 🧪 1️⃣ NLP — AG News with Garbage Web Text

| Epoch | Baseline Accuracy | CurriculumGuard Accuracy |
|------:|------------------:|-------------------------:|
| 0     | 0.64              | 0.59                     |
| 2     | 0.69              | 0.70                     |
| 5     | —                 | **0.72**                 |
| 7     | —                 | **0.739**                |

**Observation:** Baseline training plateaus early due to noisy web text. CurriculumGuard keeps improving by suppressing unstable samples.

---

### 🧪 2️⃣ Vision — FashionMNIST with 35% Label Noise

| Epoch | Baseline Accuracy | CurriculumGuard Accuracy |
|------:|------------------:|-------------------------:|
| 0     | 0.837             | **0.850**                |
| 2     | 0.840             | **0.859**                |
| 7     | —                 | **0.875**                |

**Observation:** Label noise stalls conventional training. CurriculumGuard dynamically downweights corrupted samples.

---

### 🧪 3️⃣ Fraud Detection — Credit Card Transactions

| Epoch | Baseline Recall | CurriculumGuard Recall |
|------:|----------------:|-----------------------:|
| 0     | 0.44            | **0.66**               |
| 2     | 0.86            | **0.88**               |
| 5     | —               | **0.90**               |

**Observation:** CurriculumGuard rapidly improves minority-class recall without destabilizing training.

---

### 🧪 4️⃣ Continual Learning — Distribution Shift

| Phase  | Baseline Accuracy | CurriculumGuard Accuracy      |
|--------|------------------:|------------------------------:|
| Task-A | 0.99              | 0.98                          |
| Task-B | 1.00              | **1.00 (no regression)**      |

**Observation:** Both systems adapt quickly, but CurriculumGuard enforces safety guarantees under distribution drift.

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

## 📥 Datasets Used in Benchmarks

The benchmarks above use the following publicly available datasets:

| Dataset | Domain | Source |
|---------|--------|--------|
| **AG News** | NLP | [Kaggle - AG News Classification](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) |
| **FashionMNIST** | Vision | `sklearn.datasets` (auto-downloads) |
| **Credit Card Fraud** | Fraud Detection | [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

All datasets are publicly available and free to use for research and benchmarking purposes.

---

## 📜 License

MIT