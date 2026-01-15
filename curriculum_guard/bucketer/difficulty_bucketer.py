import numpy as np

class DifficultyBucketer:
    def __init__(self, profiler):
        self.profiler = profiler

    def bucketize(self):
        scores = []

        for sid, s in self.profiler.states.items():
            if s.count < 3: continue
            score = 0.5*s.ema_loss + 0.3*s.loss_var + 0.2*s.entropy + 0.5*s.forgetting
            scores.append((sid, score))

        if len(scores) < 20: return {}

        ids, vals = zip(*scores)
        p = np.percentile(vals, [20,40,60,80])

        buckets = dict(easy=[], learnable=[], hard=[], noisy=[], harmful=[])

        for sid,v in scores:
            if v<=p[0]: buckets["easy"].append(sid)
            elif v<=p[1]: buckets["learnable"].append(sid)
            elif v<=p[2]: buckets["hard"].append(sid)
            elif v<=p[3]: buckets["noisy"].append(sid)
            else: buckets["harmful"].append(sid)
        
        if not any(len(v) > 0 for v in buckets.values()):
            return {}

        return buckets
        

