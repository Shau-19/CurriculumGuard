from curriculum_guard.core.guard import CurriculumGuard
from curriculum_guard.engine.engine import CurriculumEngine


class Curriculum:
    """
    CurriculumGuard v0.2 public API.

    Minimal, safe, and model-agnostic training-time
    data control for PyTorch.
    """

    def __init__(self, engine: CurriculumEngine):
        self._engine = engine   # internal by design

    @classmethod
    def auto(cls, dataset, **kwargs):
        """
        Create a curriculum with safe automatic defaults.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset returning (id, input, target)
        """
        guard = CurriculumGuard(dataset, **kwargs)
        engine = CurriculumEngine(dataset, guard)
        return cls(engine)

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
