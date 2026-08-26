# curriculum_guard/engine/engine.py

from torch.utils.data import DataLoader
from curriculum_guard.sampler.adaptive_sampler import AdaptiveSampler


class CurriculumEngine:
    """
    Internal execution engine for CurriculumGuard.

    Responsibilities:
    - Build curriculum-aware samplers
    - Wrap DataLoaders safely
    - Route training signals back to CurriculumGuard
    - Handle warmup epochs
    """

    def __init__(self, dataset, guard):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset returning (id, input, target)

        guard : CurriculumGuard
            Core curriculum controller
        """
        self.dataset = dataset
        self.guard = guard

        # bookkeeping
        self._step_count = 0
        self._epoch_count = 0
        self._last_stats = {}

    # -------------------------------------------------
    # DataLoader wrapping
    # -------------------------------------------------
    def wrap_loader(self, dataloader: DataLoader) -> DataLoader:
        """
        Replace the sampler of an existing DataLoader
        with a curriculum-aware sampler.

        During warmup epochs, returns original loader unchanged.
        """
        # Check if still in warmup
        if self._epoch_count < self.guard.warmup_epochs:
            print(f"[CurriculumGuard] Warmup epoch {self._epoch_count + 1}/{self.guard.warmup_epochs}")
            self._epoch_count += 1
            return dataloader
        
        # Update buckets before creating new sampler
        self.guard.buckets = self.guard.bucketer.bucketize()
        
        # If buckets are empty or insufficient, fall back to original loader
        if not self.guard.buckets or sum(len(v) for v in self.guard.buckets.values()) < len(self.dataset) // 5:
            print("[CurriculumGuard] Insufficient profiling data, using original loader")
            self._epoch_count += 1
            return dataloader
        
        sampler = AdaptiveSampler(
            dataset=self.dataset,
            guard=self.guard,
        )
        
        self._epoch_count += 1

        return DataLoader(
            self.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
        )

    # -------------------------------------------------
    # Training step hook
    # -------------------------------------------------
    def step(self, ids, loss, logits, targets):
        """
        Update curriculum state after each training step.

        Parameters
        ----------
        ids : tensor or list
            Sample IDs from the batch
        loss : tensor
            Per-sample losses (detached)
        logits : tensor
            Model outputs (detached)
        targets : tensor
            Ground truth labels
        """
        self._step_count += 1

        # Update profiler (sample-level signals)
        self.guard.profiler.update(
            ids=ids,
            losses=loss,
            logits=logits,
            labels=targets,
        )

        # Periodic curriculum updates (every 100 steps)
        # Re-bucketize and update weights based on profiling
        if self._step_count % 100 == 0:
            self._update_curriculum()

        # Cache lightweight stats
        self._last_stats = {
            "step": self._step_count,
            "epoch": self._epoch_count,
            "weights": self.guard.weights.copy(),
        }

    def _update_curriculum(self):
        """
        Periodic update of buckets and weights.
        
        This runs every N steps to refresh the curriculum
        based on accumulated profiling data.
        """
        # Re-bucketize
        new_buckets = self.guard.bucketer.bucketize()
        
        if new_buckets:
            self.guard.buckets = new_buckets

    # -------------------------------------------------
    # Read-only introspection
    # -------------------------------------------------
    def stats(self):
        """
        Safe, read-only snapshot of curriculum state.
        
        Returns
        -------
        dict
            Statistics including step count, weights, bucket sizes, etc.
        """
        stats = dict(self._last_stats)
        
        # Add bucket information
        if self.guard.buckets:
            stats["bucket_sizes"] = {
                k: len(v) for k, v in self.guard.buckets.items()
            }
        
        # Add profiler stats
        stats["samples_profiled"] = len(self.guard.profiler.states)
        
        return stats
