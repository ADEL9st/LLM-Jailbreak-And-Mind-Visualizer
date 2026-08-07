"""Root conftest — its presence puts `backend/` on sys.path, so tests can
`import app.…` the same way uvicorn does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
