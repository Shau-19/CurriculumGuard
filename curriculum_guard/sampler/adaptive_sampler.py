import random
from torch.utils.data import Sampler
class AdaptiveSampler(Sampler):
    def __init__(self, dataset, guard):
        self.dataset = dataset
        self.guard = guard

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        buckets = self.guard.buckets
        weights = self.guard.weights

        # 🔥 WARMUP FALLBACK
        if not buckets or sum(len(v) for v in buckets.values()) < len(self.dataset)//5:
            return iter(range(len(self.dataset)))

        # Filter out empty buckets to ensure we always sample valid indices
        active_buckets = {k: v for k, v in buckets.items() if len(v) > 0}
        if not active_buckets:
            return iter(range(len(self.dataset)))

        names = list(active_buckets.keys())
        probs = [weights[n] for n in names]

        # Fallback to uniform probabilities if total weight is 0
        total_prob = sum(probs)
        if total_prob == 0:
            probs = [1.0 / len(names)] * len(names)

        chosen = random.choices(names, weights=probs, k=len(self.dataset))

        idxs = []
        for b in chosen:
            idxs.append(random.choice(active_buckets[b]))

        return iter(idxs)
