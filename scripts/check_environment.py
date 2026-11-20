from __future__ import annotations

import ctypes
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import casadi
import matplotlib
import numpy
import scipy
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import acados_template  # noqa: F401

    acados_root = Path(os.environ["ACADOS_SOURCE_DIR"])
    library = "libacados.dylib" if sys.platform == "darwin" else "libacados.so"
    ctypes.CDLL(str(acados_root / "lib" / library))

    lookup = loadmat(PROJECT_ROOT / "data" / "offline" / "paper_k5_lookup.mat")
    controller_1 = lookup["Controller1"]
    controller_2 = lookup["Controller2"]
    k_value = int(lookup["K"].item())

    print(f"Python:          {sys.version.split()[0]} ({platform.machine()})")
    print(f"CasADi:          {casadi.__version__}")
    print(f"acados_template: {version('acados_template')}")
    print(f"NumPy:           {numpy.__version__}")
    print(f"SciPy:           {scipy.__version__}")
    print(f"Matplotlib:      {matplotlib.__version__}")
    print(f"Lookup table:    K={k_value}, Controller1={controller_1.shape}, Controller2={controller_2.shape}")
    print("Environment check passed.")


if __name__ == "__main__":
    main()

