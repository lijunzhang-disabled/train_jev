#!/usr/bin/env bash
# Install a pinned OSS CAD Suite build (Yosys, SymbiYosys, SMT solvers, Verilator, Icarus) without root.
#
# Usage: tools/install_oss_cad_suite.sh [install_dir]     (default: $HOME/opt)
# Then add it to PATH, e.g. in ~/.bashrc:
#   export PATH="$HOME/opt/oss-cad-suite/bin:$PATH"
#
# The file is fetched in 32 MB chunks over CONNECTIONS parallel connections (default 8), which is much
# faster when each connection is throttled. Every chunk resumes after a break, within a run and on rerun.
# If downloading keeps failing, download the .tgz on another machine, copy it into install_dir and rerun;
# the script uses that file. OSS_CAD_URL overrides the download URL (e.g. a mirror).
# Either way the file is checked against the SHA-256 that GitHub publishes for this release.
set -euo pipefail

RELEASE="2026-09-28"
ASSET="oss-cad-suite-linux-x64-${RELEASE//-/}.tgz"
SIZE=745346604
SHA256=6d04cf3e36c45d4eacbe2fa07f3860b9a86f503c2b725bdee4d02e54defbf80d
URL=${OSS_CAD_URL:-https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$RELEASE/$ASSET}
CONNECTIONS=${CONNECTIONS:-8}
CHUNK=$((32 * 1048576))   # fixed, so chunks from an earlier run line up whatever CONNECTIONS is
DEST="${1:-$HOME/opt}"
TGZ="$DEST/$ASSET"
PART="$TGZ.part"

size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

fetch_chunk() {   # fill $TGZ.chunk.NNN with chunk $1, resuming after every break; give up after 5 tries in a row without progress
    local out start end len have err stuck=0
    out=$(printf '%s.chunk.%03d' "$TGZ" "$1")
    start=$(($1 * CHUNK))
    end=$((start + CHUNK - 1))
    [ "$end" -lt "$SIZE" ] || end=$((SIZE - 1))
    len=$((end - start + 1))
    while have=$(size_of "$out"); [ "$have" -lt "$len" ]; do
        if [ "$stuck" -ge 5 ]; then
            echo "   chunk $1: no progress in 5 attempts. Last error: ${err:-none}" >&2
            return 1
        fi
        # --speed-limit/--speed-time: drop a stalled connection after 60 s below 1 KB/s, then resume
        err=$(curl -sSfL --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
              -r "$((start + have))-$end" "$URL" 2>&1 >> "$out") || true
        if [ "$(size_of "$out")" -gt "$have" ]; then stuck=0; else stuck=$((stuck + 1)); sleep 5; fi
    done
    if [ "$have" -gt "$len" ]; then
        rm -f "$out"
        echo "   chunk $1: the server ignores byte ranges, so it can't be resumed. Copy the file in instead (step 5)." >&2
        return 1
    fi
}

downloaded() { find "$DEST" -maxdepth 1 -name "$ASSET.chunk.*" -printf '%s\n' | awk '{s += $1} END {print s + 0}'; }

progress() {   # print the total every 10 s
    local start_bytes start_time=$SECONDS got
    start_bytes=$(downloaded)
    while sleep 10; do
        got=$(downloaded)
        echo "   $((got * 100 / SIZE))%  $((got / 1048576)) of $((SIZE / 1048576)) MB  $(((got - start_bytes) / 1024 / (SECONDS - start_time))) KB/s"
    done
}

download() {
    local n=$(((SIZE + CHUNK - 1) / CHUNK)) reporter
    if [ -f "$PART" ] && ! compgen -G "$TGZ.chunk.*" > /dev/null; then
        split -b "$CHUNK" -d -a 3 "$PART" "$TGZ.chunk."   # keep what an earlier single-connection download got
        rm "$PART"
    fi
    echo "downloading $ASSET ($((SIZE / 1048576)) MB) in $n chunks over $CONNECTIONS connections; $(($(downloaded) / 1048576)) MB already here"
    export URL SIZE CHUNK TGZ
    export -f fetch_chunk size_of
    progress &
    reporter=$!
    trap "kill $reporter 2>/dev/null" EXIT   # also stop it if the script is interrupted
    if ! seq 0 $((n - 1)) | xargs -P "$CONNECTIONS" -I{} bash -c 'fetch_chunk {}'; then
        echo "FAILED: some chunks did not finish; $(($(downloaded) / 1048576)) of $((SIZE / 1048576)) MB are kept in $DEST." >&2
        echo "  Rerun to continue, or copy the file from another machine (docs/phase0_guide.md, step 5)." >&2
        exit 1
    fi
    kill "$reporter" 2>/dev/null || true
    trap - EXIT
    cat "$TGZ".chunk.* > "$PART"
    mv "$PART" "$TGZ"
    rm "$TGZ".chunk.*
}

mkdir -p "$DEST"
exec 9> "$DEST/.$ASSET.lock"   # two runs at once would write into the same chunks
flock -n 9 || { echo "another install_oss_cad_suite.sh is still running for $DEST; wait for it or stop it" >&2; exit 1; }
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
