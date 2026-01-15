import torch
from collections import defaultdict

class SampleState:
    def __init__(self):
        self.ema_loss = None
        self.loss_var = 0.0
        self.forgetting = 0
        self.last_correct = None
        self.entropy = 0.0
        self.count = 0

class SampleProfiler:
    def __init__(self, decay=0.98):
        self.states = defaultdict(SampleState)
        self.decay = decay

    def _entropy(self, logits):
        p = torch.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-9)).sum(dim=-1).cpu()

    def update(self, ids, losses, logits, labels):
        ent = self._entropy(logits)
        preds = logits.argmax(dim=-1).cpu()
        correct = (preds == labels.cpu())

        for i, sid in enumerate(ids):
            s = self.states[int(sid)]
            loss = float(losses[i].detach().cpu())

            if s.ema_loss is None:
                s.ema_loss = loss
            else:
                delta = loss - s.ema_loss
                s.ema_loss = self.decay*s.ema_loss + (1-self.decay)*loss
                s.loss_var = self.decay*s.loss_var + (1-self.decay)*(delta**2)

            s.entropy = self.decay*s.entropy + (1-self.decay)*float(ent[i].detach().cpu())

            if s.last_correct is not None and s.last_correct and not correct[i]:
                s.forgetting += 1

            s.last_correct = bool(correct[i])
            s.count += 1
