from curriculum_guard.policy.curriculum_policy import CurriculumPolicy
from curriculum_guard.safety.safety_controller import SafetyController
from curriculum_guard.bucketer.difficulty_bucketer import DifficultyBucketer
from curriculum_guard.profiler.sample_profiler import SampleProfiler
from curriculum_guard.core.state import CurriculumState

class CurriculumGuard:
    def __init__(self,dataset):
        self.profiler=SampleProfiler()
        self.bucketer=DifficultyBucketer(self.profiler)
        self.safety=SafetyController()
        self.policy=CurriculumPolicy()
        self.weights={"easy":0.2,"learnable":0.4,"hard":0.25,"noisy":0.1,"harmful":0.05}
        self.prev=None

    def snapshot(self):
        return CurriculumState(self.buckets,self.weights.copy())

    def step(self,val_loss):
        if self.safety.record(val_loss):
            state=self.safety.rollback()
            self.weights=state.weights
        else:
            self.safety.mark_safe(self.snapshot())
