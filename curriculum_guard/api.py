from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.engine.engine import CurriculumEngine


class Curriculum:
    """
    CurriculumGuard v0.2 public API.

    Minimal, safe, and model-agnostic training-time
    data control for PyTorch.

    Philosophy:
    - Beginners: zero configuration
    - Practitioners: safe knobs
    - Researchers: full control
    """

    def __init__(self, engine: CurriculumEngine):
        self._engine = engine   # internal by design

    # --------------------------------------------------
    # 🟢 Beginner / Intermediate API
    # --------------------------------------------------
    @classmethod
    def auto(
        cls,
        dataset,
        *,
        sensitivity: str = "medium",
        safety: bool = True,
        warmup_epochs: int = 0,
        **kwargs,
    ):
        """
        Create a curriculum with safe automatic defaults.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset returning (id, input, target)

        sensitivity : {"low", "medium", "high"}, optional
            Controls how aggressively sample weights adapt.

        safety : bool, optional
            Enable rollback-based safety (default: True)

        warmup_epochs : int, optional
            Number of initial epochs before curriculum activates.

        Notes
        -----
        This is the recommended entry point for most users.
        """
        guard = CurriculumGuard(
            dataset,
            sensitivity=sensitivity,
            safety=safety,
            warmup_epochs=warmup_epochs,
            **kwargs,
        )
        engine = CurriculumEngine(dataset, guard)
        return cls(engine)

    # --------------------------------------------------
    # 🔵 Advanced API (explicit strategy selection)
    # --------------------------------------------------
    @classmethod
    def custom(
        cls,
        dataset,
        *,
        policy: str = "default",
        bucketing: str = "adaptive",
        safety: str = "rollback",
        entropy_weight: float = 0.2,
        **kwargs,
    ):
        """
        Create a curriculum with explicit strategy choices.

        Parameters
        ----------
        policy : str
            Built-in curriculum policy name.

        bucketing : str
            Difficulty bucketing strategy.

        safety : str
            Safety controller strategy.

        entropy_weight : float
            Weight of uncertainty in difficulty scoring.
        """
        guard = CurriculumGuard(
            dataset,
            policy=policy,
            bucketing=bucketing,
            safety_controller=safety,
            entropy_weight=entropy_weight,
            **kwargs,
        )
        engine = CurriculumEngine(dataset, guard)
        return cls(engine)

    # --------------------------------------------------
    # 🔴 Research API (full composability)
    # --------------------------------------------------
    @classmethod
    def from_components(
        cls,
        *,
        profiler,
        policy,
        bucketer,
        safety,
        dataset,
    ):
        """
        Create a curriculum from fully custom components.

        Intended for research, ablations, and experimentation.
        """
        guard = CurriculumGuard.from_components(
            dataset=dataset,
            profiler=profiler,
            policy=policy,
            bucketer=bucketer,
            safety=safety,
        )
        engine = CurriculumEngine(dataset, guard)
        return cls(engine)

    # --------------------------------------------------
    # Runtime API
    # --------------------------------------------------
    def __call__(self, dataloader):
        """
        Wrap a DataLoader with curriculum-aware sampling.

        Usage:
            for batch in curriculum(loader):
                ...
        """
        return self._engine.wrap_loader(dataloader)

    def step(self, ids, loss, logits, targets):
        """
        Update curriculum state.

        Must be called once per training step.
        """
        self._engine.step(ids, loss, logits, targets)

    def stats(self):
        """
        Return read-only curriculum statistics.
        """
        return self._engine.stats()
