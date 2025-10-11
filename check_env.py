#!/usr/bin/env python3
"""Quick sanity check that PyTorch and CUDA are visible in the current env."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - straightforward reporting
        print("PyTorch import failed. Install torch with CUDA support first.", file=sys.stderr)
        print(f"ImportError: {exc}", file=sys.stderr)
        return 1

    print(f"torch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"cuda available: {cuda_available}")

    if not cuda_available:
        print("No CUDA device detected. Check driver, CUDA toolkit, or torch build.")
        return 0

    device_count = torch.cuda.device_count()
    print(f"cuda device count: {device_count}")

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        print(f" - device {idx}: {name} (compute capability {capability[0]}.{capability[1]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
