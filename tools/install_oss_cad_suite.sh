#!/usr/bin/env bash
# Install a pinned OSS CAD Suite build (Yosys, SymbiYosys, SMT solvers, Verilator, Icarus) without root.
#
# Usage: tools/install_oss_cad_suite.sh [install_dir]     (default: $HOME/opt)
# Then add it to PATH, e.g. in ~/.bashrc:
#   export PATH="$HOME/opt/oss-cad-suite/bin:$PATH"
set -euo pipefail

RELEASE="2026-09-28"
ASSET="oss-cad-suite-linux-x64-${RELEASE//-/}.tgz"
URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$RELEASE/$ASSET"
DEST="${1:-$HOME/opt}"

mkdir -p "$DEST"
if [ -x "$DEST/oss-cad-suite/bin/yosys" ]; then
    echo "already installed: $DEST/oss-cad-suite"
else
    echo "downloading $ASSET (~750 MB)"
    curl -fL --retry 3 --progress-bar -o "$DEST/$ASSET.part" "$URL"
    mv "$DEST/$ASSET.part" "$DEST/$ASSET"
    tar -xzf "$DEST/$ASSET" -C "$DEST"
    rm "$DEST/$ASSET"
fi
"$DEST/oss-cad-suite/bin/yosys" -V
"$DEST/oss-cad-suite/bin/sby" --help >/dev/null && echo "sby OK"
echo
echo "add to PATH:  export PATH=\"$DEST/oss-cad-suite/bin:\$PATH\""
