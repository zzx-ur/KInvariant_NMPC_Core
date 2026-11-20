# Installation

Use a macOS or Linux terminal with Conda, Git, CMake, Make, and a C compiler
installed. On macOS, the Xcode Command Line Tools provide the compiler and Make.
The reference simulation was tested on Apple Silicon with Python 3.10,
CasADi 3.7.2, and acados_template 0.5.1. The included lookup table makes MATLAB
unnecessary for this example. Choose an acados installation path without spaces
so the generated Makefile can resolve its library and header paths.

## 1. Create the Python environment

Open a terminal in the project folder.

```sh
conda env create -n kinv-core -f environment.yml
conda activate kinv-core
```

## 2. Build and connect acados

Run this once to install the tested acados revision in your home directory.

```sh
export ACADOS_SOURCE_DIR="$HOME/acados"
git clone https://github.com/acados/acados.git "$ACADOS_SOURCE_DIR"
git -C "$ACADOS_SOURCE_DIR" checkout 8c09983bd57951b0f6ab172120369f959e86de39
git -C "$ACADOS_SOURCE_DIR" submodule update --init --recursive

cmake -S "$ACADOS_SOURCE_DIR" -B "$ACADOS_SOURCE_DIR/build" \
  -DCMAKE_INSTALL_PREFIX="$ACADOS_SOURCE_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DBLASFEO_TARGET=GENERIC
cmake --build "$ACADOS_SOURCE_DIR/build" --target install --parallel 4
```

If you already built the same acados revision, set `ACADOS_SOURCE_DIR` to the
root of that installation and reuse it. In either case, install the matching
Python interface in the active Conda environment.

```sh
python -m pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
```

## 3. Check and run

```sh
sh scripts/run_python.sh scripts/check_environment.py
sh scripts/run_python.sh scripts/simulate_hard_reference.py --steps 50
```

The simulation generates its solver on startup. On the first run, acados may
download its template renderer. The trajectory plot is saved to
`results/hard_reference_trajectory.png`.

In a new terminal, activate `kinv-core` and set `ACADOS_SOURCE_DIR` again before
running. For platform-specific build help, see the
[acados installation guide](https://docs.acados.org/installation/index.html) and
[Python interface setup](https://docs.acados.org/python_interface/index.html#installation).
