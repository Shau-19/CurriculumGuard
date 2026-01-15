import random
from torch.utils.data import Sampler
class AdaptiveSampler(Sampler):
    def __init__(self, dataset, buckets, weights):
        self.dataset = dataset
        self.buckets = buckets
        self.weights = weights

    def __iter__(self):

        # 🔥 WARMUP FALLBACK
        if not self.buckets or sum(len(v) for v in self.buckets.values()) < len(self.dataset)//5:
            return iter(range(len(self.dataset)))

        names = list(self.buckets.keys())
        probs = [self.weights[n] for n in names]

        chosen = random.choices(names, probs, k=len(self.dataset))

        idxs = []
        for b in chosen:
            if self.buckets[b]:
                idxs.append(random.choice(self.buckets[b]))

        return iter(idxs)
