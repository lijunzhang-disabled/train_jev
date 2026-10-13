#!/usr/bin/env bash
# Install a pinned OSS CAD Suite build (Yosys, SymbiYosys, SMT solvers, Verilator, Icarus) without root.
#
# Usage: tools/install_oss_cad_suite.sh [install_dir]     (default: $HOME/opt)
# Then add it to PATH, e.g. in ~/.bashrc:
#   export PATH="$HOME/opt/oss-cad-suite/bin:$PATH"
#
# The download resumes where it stopped, both within a run and when the script is rerun.
# If it keeps failing, download the .tgz on another machine, copy it into install_dir and rerun;
# the script uses that file. OSS_CAD_URL overrides the download URL (e.g. a mirror).
# Either way the file is checked against the SHA-256 that GitHub publishes for this release.
set -euo pipefail

RELEASE="2026-09-28"
ASSET="oss-cad-suite-linux-x64-${RELEASE//-/}.tgz"
SIZE=745346604
SHA256=6d04cf3e36c45d4eacbe2fa07f3860b9a86f503c2b725bdee4d02e54defbf80d
URL=${OSS_CAD_URL:-https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$RELEASE/$ASSET}
DEST="${1:-$HOME/opt}"
TGZ="$DEST/$ASSET"
PART="$TGZ.part"

size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

download() {   # fill $PART up to $SIZE bytes, resuming after every break; give up after 5 tries in a row without progress
    local have stuck=0
    echo "downloading $ASSET ($((SIZE / 1048576)) MB)"
    while have=$(size_of "$PART"); [ "$have" -lt "$SIZE" ]; do
        if [ "$stuck" -ge 5 ]; then
            echo "FAILED: no progress in 5 attempts; $((have / 1048576)) of $((SIZE / 1048576)) MB are kept in $PART." >&2
            echo "  Rerun to continue, or copy the file from another machine (docs/phase0_guide.md, step 5)." >&2
            exit 1
        fi
        [ "$have" -gt 0 ] && echo "   resuming at $((have / 1048576)) MB"
        # --speed-limit/--speed-time: drop a stalled connection after 60 s below 1 KB/s, then resume
        curl -fL -C - --connect-timeout 30 --speed-limit 1024 --speed-time 60 --progress-bar -o "$PART" "$URL" || true
        if [ "$(size_of "$PART")" -gt "$have" ]; then stuck=0; else stuck=$((stuck + 1)); sleep 5; fi
    done
    mv "$PART" "$TGZ"
}

mkdir -p "$DEST"
if [ -x "$DEST/oss-cad-suite/bin/yosys" ]; then
    echo "already installed: $DEST/oss-cad-suite"
else
    [ -f "$TGZ" ] && echo "using $TGZ" || download
    if ! echo "$SHA256  $TGZ" | sha256sum --check --quiet; then
        rm -f "$TGZ"
        echo "FAILED: $ASSET is corrupt (SHA-256 mismatch) and was deleted; rerun to download it again." >&2
        exit 1
    fi
    echo "SHA-256 OK; unpacking"
    tar -xzf "$TGZ" -C "$DEST"
    rm "$TGZ"
fi
"$DEST/oss-cad-suite/bin/yosys" -V
"$DEST/oss-cad-suite/bin/sby" --help >/dev/null && echo "sby OK"
echo
echo "add to PATH:  export PATH=\"$DEST/oss-cad-suite/bin:\$PATH\""
