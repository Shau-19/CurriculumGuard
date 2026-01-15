import copy

class CurriculumPolicy:
    def propose(self, w, fb):
        n = copy.deepcopy(w)
        if fb["val_delta"]>0:
            n["easy"]*=1.03
            n["harmful"]*=0.9
        else:
            n["hard"]*=1.02
        s=sum(n.values())
        for k in n: n[k]/=s
        return n
