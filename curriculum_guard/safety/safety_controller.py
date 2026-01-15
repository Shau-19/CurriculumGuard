from collections import deque

class SafetyController:
    def __init__(self, patience=4, tol=0.03):
        self.hist = deque(maxlen=10)
        self.bad = 0
        self.patience = patience
        self.tol = tol
        self.safe_state = None

    def record(self, val_loss):
        self.hist.append(val_loss)
        if len(self.hist)<5: return False
        if self.hist[-1]-self.hist[-5] > self.tol:
            self.bad +=1
        else:
            self.bad =0
        return self.bad>=self.patience

    def mark_safe(self, state):
        self.safe_state = state

    def rollback(self):
        return self.safe_state
