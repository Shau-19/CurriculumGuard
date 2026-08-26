
from curriculum_guard.policy.curriculum_policy import CurriculumPolicy
from curriculum_guard.safety.safety_controller import SafetyController
from curriculum_guard.bucketer.difficulty_bucketer import DifficultyBucketer
from curriculum_guard.profiler.sample_profiler import SampleProfiler
from curriculum_guard.core.state import CurriculumState


class CurriculumGuard:
    
    
    def __init__(
        self,
        dataset,
        sensitivity="medium",
        safety=True,
        warmup_epochs=0,
        policy=None,
        bucketing=None,
        safety_controller=None,
        entropy_weight=0.2,
        **kwargs
    ):
        
        self.dataset = dataset
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Initialize profiler (always the same for now)
        if isinstance(policy, SampleProfiler):
            self.profiler = policy  # Custom profiler passed in
        else:
            # Set decay based on sensitivity
            decay_map = {"low": 0.95, "medium": 0.98, "high": 0.99}
            decay = decay_map.get(sensitivity, 0.98)
            self.profiler = SampleProfiler(decay=decay)
        
        # Initialize bucketer
        if hasattr(bucketing, 'bucketize'):
            # Custom bucketer passed in
            self.bucketer = bucketing
        else:
            # Use default DifficultyBucketer
            self.bucketer = DifficultyBucketer(self.profiler)
        
        # Initialize safety controller
        if hasattr(safety_controller, 'record'):
            # Custom safety controller passed in
            self.safety = safety_controller
        elif safety_controller == "rollback" or safety:
            # Use default safety controller
            patience_map = {"low": 6, "medium": 4, "high": 2}
            patience = patience_map.get(sensitivity, 4)
            self.safety = SafetyController(patience=patience)
        else:
            # No safety controller
            self.safety = None
        
        # Initialize policy
        if hasattr(policy, 'propose'):
            # Custom policy passed in
            self.policy = policy
        else:
            # Use default policy (could extend with policy="anti_noise", etc.)
            self.policy = CurriculumPolicy()
        
        # Initialize weights and buckets
        self.weights = {
            "easy": 0.2,
            "learnable": 0.4,
            "hard": 0.25,
            "noisy": 0.1,
            "harmful": 0.05
        }
        self.buckets = {}
        self.prev = None
        self.prev_val_loss = None
    
    @classmethod
    def from_components(cls, dataset, profiler, policy, bucketer, safety):
        """
        Create CurriculumGuard from fully custom components.
        
        For research and experimentation.
        """
        guard = cls.__new__(cls)
        guard.dataset = dataset
        guard.profiler = profiler
        guard.bucketer = bucketer
        guard.safety = safety
        guard.policy = policy
        guard.warmup_epochs = 0
        guard.current_epoch = 0
        guard.weights = {
            "easy": 0.2,
            "learnable": 0.4,
            "hard": 0.25,
            "noisy": 0.1,
            "harmful": 0.05
        }
        guard.buckets = {}
        guard.prev = None
        guard.prev_val_loss = None
        return guard
    
    def snapshot(self):
        """Create a snapshot of current curriculum state."""
        return CurriculumState(self.buckets, self.weights.copy())
    
    def step(self, val_loss):
        # Calculate validation delta
        if self.prev_val_loss is not None:
            val_delta = val_loss - self.prev_val_loss
        else:
            val_delta = 0.0
        self.prev_val_loss = val_loss

        if self.safety is None:
            # No safety controller, just update weights using policy
            fb = {"val_delta": val_delta}
            self.weights = self.policy.propose(self.weights, fb)
            return
        
        if self.safety.record(val_loss):
            # Rollback to safe state
            state = self.safety.rollback()
            if state:
                self.weights = state.weights
                self.buckets = state.buckets
        else:
            # Mark current state as safe
            self.safety.mark_safe(self.snapshot())
            # Update weights using policy since it's safe!
            fb = {"val_delta": val_delta}
            self.weights = self.policy.propose(self.weights, fb)
