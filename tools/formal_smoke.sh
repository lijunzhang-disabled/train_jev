#!/usr/bin/env bash
# SymbiYosys smoke test: one property that must be proved, one injected bug that must be caught.
# Needs sby and yosys on PATH (tools/install_oss_cad_suite.sh).
#
# Usage: tools/formal_smoke.sh [work_dir]     (default: a fresh temp dir)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK"
cp "$HERE"/formal_smoke/* "$WORK"/
cd "$WORK" || exit 1

for tool in yosys sby; do
    command -v "$tool" >/dev/null || { echo "FAIL: $tool not on PATH"; exit 1; }
done

# bmc_bug is meant to FAIL, which makes sby exit non-zero; read each task's status file instead.
sby -f smoke.sby > sby.log 2>&1
ok=$(cut -d' ' -f1 smoke_prove_ok/status 2>/dev/null || echo MISSING)
bug=$(cut -d' ' -f1 smoke_bmc_bug/status 2>/dev/null || echo MISSING)
echo "prove_ok: $ok   (expected PASS: the property is proved for all time)"
echo "bmc_bug:  $bug   (expected FAIL: the injected bug is caught)"
if [ "$bug" = FAIL ]; then
    trace=$(ls smoke_bmc_bug/engine_0/*.vcd 2>/dev/null | head -1)
    [ -n "$trace" ] && echo "          counterexample trace: $WORK/$trace"
fi

# yosys-slang gives Yosys a much fuller SystemVerilog frontend; Phase 1 needs it, Phase 0 doesn't.
if yosys -m slang -p "read_slang counter.sv; hierarchy -top counter" > slang.log 2>&1; then
    echo "yosys-slang: loaded"
else
    echo "yosys-slang: NOT available (needed from Phase 1; see $WORK/slang.log)"
fi

if [ "$ok" = PASS ] && [ "$bug" = FAIL ]; then
    echo "formal smoke test: OK"
else
    echo "formal smoke test: FAILED (log: $WORK/sby.log)"
    exit 1
fi
