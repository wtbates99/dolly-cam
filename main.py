from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


from dolly_cam.app import main


if __name__ == "__main__":
    raise SystemExit(main())
