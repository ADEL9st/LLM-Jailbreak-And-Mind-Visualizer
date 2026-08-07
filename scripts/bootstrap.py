#!/usr/bin/env python3
"""Prepare the local runtime for LLM Mind Visualizer.

The launcher calls this file before starting either server.  It keeps the
normal first-run experience simple while selecting a CPU PyTorch wheel on
machines without an NVIDIA driver.  Existing environments are reused, so a
second launch does not reinstall anything.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "backend" / ".venv"


def _run(command: list[str], *, cwd: Path | None = None, optional: bool = False) -> bool:
    print("+", " ".join(command))
    try:
        subprocess.run(command, cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError:
        if optional:
            print("Optional package installation failed; continuing.")
            return False
        raise


def _venv_python() -> Path:
    name = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV / name


def _has_nvidia() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        subprocess.run(
            [executable, "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _imports_work(python: Path, *modules: str) -> bool:
    probe = "import " + ", ".join(modules)
    return subprocess.run(
        [str(python), "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _install_ml(python: Path, gpu: bool) -> None:
    requirements = ROOT / "backend" / "requirements-ml.txt"
    lines = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    torch_specs = [line for line in lines if line.startswith(("torch=", "torch<", "torch>", "torch["))]
    vision_specs = [line for line in lines if line.startswith("torchvision")]
    other_specs = [line for line in lines if line not in torch_specs + vision_specs]

    if gpu:
        print("NVIDIA GPU detected; installing the standard PyTorch wheel set.")
        _run([str(python), "-m", "pip", "install", "-r", str(requirements)])
    else:
        print("No NVIDIA GPU detected; installing CPU-only PyTorch wheels.")
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                *torch_specs,
                *vision_specs,
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        _run([str(python), "-m", "pip", "install", *other_specs])


def prepare(*, skip_ml: bool = False, skip_nnsight: bool = False) -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")

    VENV.parent.mkdir(parents=True, exist_ok=True)
    python = _venv_python()
    if not python.exists():
        print("Creating backend virtual environment...")
        _run([sys.executable, "-m", "venv", str(VENV)])

    # Do not upgrade or replace a working environment on every launch.  This
    # is deliberately a compatibility check, not a blanket reinstall.
    if not _imports_work(python, "fastapi", "uvicorn", "pydantic"):
        print("Installing backend dependencies...")
        _run([str(python), "-m", "pip", "install", "-r", str(ROOT / "backend" / "requirements.txt")])
    else:
        print("Existing backend packages are compatible; keeping them.")

    if not skip_ml and not _imports_work(python, "torch", "transformers"):
        _install_ml(python, gpu=_has_nvidia())
    elif not skip_ml:
        print("Existing ML packages are compatible; keeping them.")

    if not skip_nnsight and not _imports_work(python, "nnsight"):
        _run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT / "backend" / "requirements-nnsight.txt")],
            optional=True,
        )

    (ROOT / "models").mkdir(exist_ok=True)
    print(f"Ready on {platform.system()} ({'NVIDIA GPU' if _has_nvidia() else 'CPU'} profile).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the dependencies needed by LLM Mind Visualizer.")
    parser.add_argument("--skip-ml", action="store_true", help="Install only the API and frontend dependencies.")
    parser.add_argument("--skip-nnsight", action="store_true", help="Skip the optional nnsight adapter.")
    args = parser.parse_args()
    prepare(
        skip_ml=args.skip_ml or os.getenv("LLM_MIND_SKIP_ML") == "1",
        skip_nnsight=args.skip_nnsight or os.getenv("LLM_MIND_SKIP_NNSIGHT") == "1",
    )


if __name__ == "__main__":
    main()
