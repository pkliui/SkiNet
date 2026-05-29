import os
import warnings

# PyTorch calls getpwuid() at import time to build its cache path; containers
# without a /etc/passwd entry for the running uid crash and leave torch._dynamo
# partially initialised, which cascades into "precompile already registered"
# errors in every subsequent test file that imports lightning/timm/smp.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")

# Suppress known-benign Lightning/LitLogger runtime warnings so they don't pollute test output.
# These are all third-party library issues, not bugs in SkiNet code.

# LitLogger does not implement log_graph(); Lightning calls it on every logger.
warnings.filterwarnings("ignore", message=".*does not support.*log_graph", category=UserWarning)

# Lightning's internal _pytree collation uses a deprecated PyTorch API for dict batches.
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*LeafSpec.*deprecated")

# Lightning infers batch_size from the first dict-batch before training_step; batch_size is
# already passed explicitly in all self.log() calls so this inference is harmless.
warnings.filterwarnings("ignore", message=".*Trying to infer the.*batch_size", category=UserWarning)
