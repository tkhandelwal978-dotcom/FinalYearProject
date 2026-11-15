"""FireGuard AI backend package."""

import sys
from pathlib import Path

# The backend reuses the `ml` package (preprocessing + predictor) from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
