"""
Quick validation script for CurriculumGuard v0.2.1

Tests all 3 API levels to ensure they work correctly.
"""
import sys,os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from curriculum_guard import Curriculum
    print("[OK] Import successful: from curriculum_guard import Curriculum")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    print("\nMake sure curriculum_guard/ is in your Python path or installed")
    sys.exit(1)

# Create a dummy dataset
class DummyDataset:
    def __len__(self):
        return 100
    
    def __getitem__(self, i):
        import torch
        return i, torch.randn(10), torch.randint(0, 2, (1,)).item()

dataset = DummyDataset()

print("\n" + "="*60)
print("Testing Beginner API")
print("="*60)

try:
    curriculum = Curriculum.auto(dataset)
    print("[OK] Curriculum.auto(dataset) works")
except Exception as e:
    print(f"[FAIL] Curriculum.auto() failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing Intermediate API")
print("="*60)

try:
    curriculum = Curriculum.auto(
        dataset,
        sensitivity="medium",
        warmup_epochs=2,
        safety=True
    )
    print("[OK] Curriculum.auto(dataset, sensitivity=...) works")
except Exception as e:
    print(f"[FAIL] Curriculum.auto() with parameters failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing Advanced API")
print("="*60)

try:
    curriculum = Curriculum.custom(
        dataset,
        policy="default",
        bucketing="adaptive",
        safety="rollback",
        entropy_weight=0.3
    )
    print("[OK] Curriculum.custom(dataset, ...) works")
except Exception as e:
    print(f"[FAIL] Curriculum.custom() failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing Research API")
print("="*60)

try:
    from curriculum_guard.profiler.sample_profiler import SampleProfiler
    from curriculum_guard.policy.curriculum_policy import CurriculumPolicy
    from curriculum_guard.bucketer.difficulty_bucketer import DifficultyBucketer
    from curriculum_guard.safety.safety_controller import SafetyController
    
    profiler = SampleProfiler()
    bucketer = DifficultyBucketer(profiler)
    
    curriculum = Curriculum.from_components(
        profiler=profiler,
        policy=CurriculumPolicy(),
        safety=SafetyController(),
        bucketer=bucketer,
        dataset=dataset
    )
    print("[OK] Curriculum.from_components(...) works")
except Exception as e:
    print(f"[FAIL] Curriculum.from_components() failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing Runtime API")
print("="*60)

try:
    import torch
    from torch.utils.data import DataLoader
    
    curriculum = Curriculum.auto(dataset)
    loader = DataLoader(dataset, batch_size=32)
    
    # Test wrap_loader
    curriculum_loader = curriculum(loader)
    print("[OK] curriculum(loader) works")
    
    # Test one batch
    for ids, x, y in curriculum_loader:
        # Test step
        fake_loss = torch.randn(len(ids))
        fake_logits = torch.randn(len(ids), 2)
        fake_targets = torch.randint(0, 2, (len(ids),))
        
        curriculum.step(ids, fake_loss, fake_logits, fake_targets)
        print("[OK] curriculum.step() works")
        
        # Test stats
        stats = curriculum.stats()
        print(f"[OK] curriculum.stats() works: {stats}")
        break
    
except Exception as e:
    print(f"[FAIL] Runtime API failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[STATUS] All tests complete!")
print("="*60)
print("\nIf all tests passed, your CurriculumGuard v0.2.1 is ready!")
