#!/bin/sh
set -eu

if [ -z "${ACADOS_SOURCE_DIR:-}" ]; then
    echo 'Set ACADOS_SOURCE_DIR to your acados installation root (see ENVIRONMENT.md).' >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) library=libacados.dylib ;;
    Linux) library=libacados.so ;;
    *) echo 'This runner supports macOS and Linux.' >&2; exit 1 ;;
esac

acados_lib="$ACADOS_SOURCE_DIR/lib"
if [ ! -f "$acados_lib/$library" ]; then
    echo "acados library not found: $acados_lib/$library" >&2
    exit 1
fi

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cache_root="$project_root/.cache"
mkdir -p "$cache_root/matplotlib"
export XDG_CACHE_HOME="$cache_root"
export MPLCONFIGDIR="$cache_root/matplotlib"

if [ "$library" = libacados.dylib ]; then
    export DYLD_LIBRARY_PATH="$acados_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$acados_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python "$@"
