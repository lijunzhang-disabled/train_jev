#!/usr/bin/env python3
"""Check the training environment (step 3 of docs/phase0_guide.md). Prints PASS / WARN / FAIL per check.

    python tools/check_env.py                       # all visible GPUs
    CUDA_VISIBLE_DEVICES=0,1 python tools/check_env.py --models Qwen/Qwen3-0.6B Qwen/Qwen3-8B

Exits non-zero if any check FAILs.
"""
import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H20_BF16_PEAK_TFLOPS = 148
failures = []


def report(status, what, detail=""):
    if status == "FAIL":
        failures.append(what)
    print(f"[{status}] {what}" + (f": {detail}" if detail else ""), flush=True)


def pinned_versions():
    pins = {}
    for line in open(os.path.join(ROOT, "requirements.txt")):
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("-", "_")] = m.group(2)
    return pins


def check_packages():
    pins = pinned_versions()
    ok = True
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "huggingface_hub"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            report("FAIL", f"import {name}", "not installed (pip install -r requirements.txt)")
            ok = False
            continue
        version, want = mod.__version__, pins.get(name)
        if want and version.split("+")[0] != want:
            report("WARN", f"{name} {version}", f"requirements.txt pins {want}")
        else:
            report("PASS", f"{name} {version}")
    try:
        import flash_attn
        report("PASS", f"flash_attn {flash_attn.__version__} (optional; use --attn flash_attention_2)")
    except ImportError:
        report("PASS", "flash_attn not installed (optional; PyTorch SDPA is used)")
    return ok


def check_gpus(torch):
    if not torch.cuda.is_available():
        report("FAIL", "CUDA", "torch.cuda.is_available() is False; check the driver and the torch wheel")
        return 0
    report("PASS", "CUDA", f"torch built for CUDA {torch.version.cuda}, NCCL "
                           f"{'.'.join(map(str, torch.cuda.nccl.version()))}")
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        report("PASS", f"GPU {i}", f"{p.name}, {p.total_memory / 2**30:.0f} GiB, compute {p.major}.{p.minor}")
    if not torch.cuda.is_bf16_supported():
        report("FAIL", "bf16", "not supported on this GPU")
    return n


def check_matmul(torch, n):
    size, iters = 8192, 30
    for i in range(n):
        dev = torch.device("cuda", i)
        a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize(dev)
        t0 = time.time()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize(dev)
        tflops = 2 * size ** 3 * iters / (time.time() - t0) / 1e12
        status = "PASS" if tflops > 0.5 * H20_BF16_PEAK_TFLOPS else "WARN"
        report(status, f"GPU {i} bf16 matmul", f"{tflops:.0f} TFLOPS (H20 dense peak ~{H20_BF16_PEAK_TFLOPS})")
        del a, b
        torch.cuda.empty_cache()


def check_topology():
    if not shutil.which("nvidia-smi"):
        report("WARN", "nvidia-smi", "not found")
        return
    out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    rows = [line for line in out.splitlines() if line.startswith(("GPU", "\tGPU", " "))][:12]
    print("    nvidia-smi topo -m (NV# = NVLink, PIX/PXB/PHB/NODE/SYS = PCIe paths):")
    for line in rows:
        print("    " + line.rstrip())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    note = f" (CUDA_VISIBLE_DEVICES={visible})" if visible else ""
    report("PASS", "topology printed", "the cards you train on should be connected by NV# entries" + note)


def nccl_worker(rank, world, port, size_mb, results):
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    x = torch.ones(size_mb * 2 ** 20 // 4, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    seconds = (time.time() - t0) / iters
    if rank == 0:
        bus = 2 * (world - 1) / world * x.numel() * 4 / seconds / 1e9  # NCCL-tests "bus bandwidth"
        results.put(bus)
    dist.destroy_process_group()


def check_nccl(n):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = ctx.SimpleQueue()
    try:
        mp.start_processes(nccl_worker, args=(n, 29517, 256, results), nprocs=n, start_method="spawn")
        bus = results.get()
        status = "PASS" if bus > 100 else "WARN"
        hint = "" if bus > 100 else " (low: are these GPUs NVLink-connected? see topology above)"
        report(status, f"NCCL all-reduce across {n} GPUs", f"bus bandwidth {bus:.0f} GB/s{hint}")
    except Exception as e:  # noqa: BLE001
        report("FAIL", f"NCCL all-reduce across {n} GPUs", repr(e))


def check_models(models):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return
    for m in models:
        try:
            path = snapshot_download(m, local_files_only=True)
            report("PASS", f"model {m}", path)
        except Exception:  # noqa: BLE001
            report("WARN", f"model {m}", "not downloaded yet (tools/download_models.sh)")


def check_disk():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for label, path in (("HF cache", hf_home), ("repo", ROOT)):
        probe = path
        while not os.path.exists(probe):
            probe = os.path.dirname(probe)
        free = shutil.disk_usage(probe).free / 2 ** 30
        report("PASS" if free > 100 else "WARN", f"free disk for {label}", f"{free:.0f} GiB at {probe}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"])
    ap.add_argument("--skip-bench", action="store_true", help="skip the matmul and NCCL benchmarks")
    args = ap.parse_args()

    report("PASS", f"python {sys.version.split()[0]}")
    if check_packages():
        import torch
        n = check_gpus(torch)
        if n and not args.skip_bench:
            check_matmul(torch, n)
        check_topology()
        if n >= 2 and not args.skip_bench:
            check_nccl(n)
    check_models(args.models)
    check_disk()
    print()
    print("environment check: " + ("OK" if not failures else f"{len(failures)} FAILED: {', '.join(failures)}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
