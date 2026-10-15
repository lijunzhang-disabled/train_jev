# Phase 0 Setup Guide: H20 Server

As of Oct 13, 2026. Git copy of the Claude Doc at
https://claude.ai/code/artifact/95cbaea8-5860-4ab6-b94c-7f0dddbb66aa (Claude updates both together).

## Overview

Phase 0 proves that every piece works on the H20 server before we scale up data: the Python environment, the GPUs, the formal tools, the teacher endpoint, and one small training run end to end. Plan for about half a day; most of it is waiting for downloads.

Run the steps in order. Each one ends with a check, and the section after step 9 lists what to paste back to me.

1. Get the code (5 min)
2. Python environment (15–30 min)
3. Check GPUs and libraries (5 min)
4. Download the models (20–60 min, depends on bandwidth)
5. Formal tools and the SymbiYosys smoke test (15 min)
6. Teacher endpoint check (10 min)
7. Build the benchmarks and the Phase 0 dataset (5 min)
8. GPU smoke run on 2 cards (10 min)
9. First real run: zero-shot baseline, then Qwen3-8B LoRA (1–2 h)

The guide assumes Linux, a working NVIDIA driver and CUDA, Python 3.10–3.13, git, and internet access to GitHub, PyPI, and ModelScope or Hugging Face. Commands run from the repo root unless a step says otherwise.

## Step 1: Get the code

Clone the repo on the server. Every later step runs from its root.

```bash
git clone git@github.com:lijunzhang-disabled/train_jev.git
cd train_jev
git log --oneline -1
```

If SSH to GitHub is blocked, clone over HTTPS with a personal access token instead. To pick up later changes, run `git pull`.

**Check:** `git log` shows the latest commit.

## Step 2: Python environment

Create a virtual environment in the repo, install the PyTorch build that matches your driver, then the pinned libraries. These are the exact versions the CPU smoke test passed with.

```bash
nvidia-smi                 # note "CUDA Version" in the top-right of the header
python3 --version          # needs 3.10 to 3.13
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch with **one** of these, matching the CUDA Version that `nvidia-smi` showed:

| nvidia-smi "CUDA Version" | Command |
| --- | --- |
| 13.0 or newer | `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130` |
| 12.6 to 12.9 | `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126` |
| older than 12.6 | Update the driver, or tell me and I'll pin an older PyTorch |

Then:

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

- The PyTorch wheel brings its own CUDA runtime, so only the driver version matters. The installed CUDA toolkit isn't used.
- You don't need flash-attn. PyTorch's built-in attention (SDPA) already uses flash kernels on H20.
- Run `source .venv/bin/activate` in every new shell before the later steps.

**Check:** the last command prints `2.14.0+cu126` (or `+cu130`), `True`, and your GPU count.

## Step 3: Check GPUs, NVLink and libraries

`tools/check_env.py` checks the library versions against `requirements.txt` and lists each GPU. It times a bf16 matrix multiply per card, prints the NVLink topology, and measures NCCL all-reduce bandwidth across the cards. Restrict it to the cards you plan to train on:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/check_env.py
```

| Line | What good looks like |
| --- | --- |
| `GPU i` | `NVIDIA H20`, about 95 GiB (or 140 GiB), compute 9.0 |
| `bf16 matmul` | roughly 100–148 TFLOPS per card; well below that means throttling or a busy card |
| topology table | `NV18` (or another `NV#`) between your cards. `PIX`/`PHB`/`SYS` means PCIe |
| `NCCL all-reduce` | well above 100 GB/s over NVLink; tens of GB/s means the cards talk over PCIe |
| `model ...` | `WARN not downloaded yet` is expected until step 4 |

**Check:** the last line reads `environment check: OK`. If the cards you picked aren't NVLink-connected, choose a different set from the topology table and rerun.

## Step 4: Download the models

Three Qwen3 checkpoints, about 22 GB in total, go into `models/` in the repo, which git ignores: 0.6B for smoke runs, 1.7B for fast debugging, and 8B for real runs. Every later script loads them from there. They come from ModelScope, whose Qwen3 files are byte-identical to Hugging Face's: I compared the SHA-256 of every weight, tokenizer and config file.

```bash
tools/download_models.sh
python tools/check_env.py --skip-bench
```

- To keep the weights on a bigger disk, run `ln -s /big/disk/models models` before downloading.
- Re-running is safe: finished files are skipped and partial ones resume. The script retries each model 3 times, then checks every weight, tokenizer and config file against the SHA-256 hashes in `tools/model_sha256.txt`.
- If the proxy fails here too (`ProxyError`, `504`), try these in order:
  - Bypass the proxy: `env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY tools/download_models.sh`
  - Use Hugging Face: `SOURCE=hf tools/download_models.sh`
  - Use a Hugging Face mirror: `SOURCE=hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 tools/download_models.sh`
- The launch scripts set `HF_HUB_OFFLINE=1`, so training and evaluation never contact Hugging Face.

If the download fails with `CERTIFICATE_VERIFY_FAILED ... self-signed certificate in certificate chain`, the proxy re-signs HTTPS traffic with its own certificate. The system's certificate store usually trusts that certificate, but Python libraries ship their own list. Point them at the system store once; step 7 needs it too:

```bash
CA=$(ls /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt 2>/dev/null | head -1); echo "$CA"
echo "export REQUESTS_CA_BUNDLE=$CA SSL_CERT_FILE=$CA" >> ~/.bashrc   # conda: conda env config vars set REQUESTS_CA_BUNDLE=$CA SSL_CERT_FILE=$CA
source ~/.bashrc
```

`REQUESTS_CA_BUNDLE` covers ModelScope; `SSL_CERT_FILE` covers Hugging Face and the GitHub downloads in step 7. If `curl https://modelscope.cn` fails with a certificate error as well, the proxy's certificate isn't installed on the system. Use the file that `pip config list` shows as `cert`, or ask IT for it.

**Check:** the three `model models/Qwen3-...` lines now read PASS.

## Step 5: Formal tools and the SymbiYosys smoke test

Install OSS CAD Suite (Yosys, SymbiYosys, SMT solvers, Verilator, Icarus) without root, pinned to the 2026-09-28 build. Then run a smoke test with two jobs: a correct counter whose property must be **proved**, and a buggy copy whose bug must be **caught**. Install it on the machine that will run the data pipeline; it only needs CPUs.

```bash
tools/install_oss_cad_suite.sh ~/opt
echo 'export PATH="$HOME/opt/oss-cad-suite/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/opt/oss-cad-suite/bin:$PATH"
tools/formal_smoke.sh
```

Expected output (from my run of the same build):

```
prove_ok: PASS   (expected PASS: the property is proved for all time)
bmc_bug:  FAIL   (expected FAIL: the injected bug is caught)
yosys-slang: loaded
formal smoke test: OK
```

- The download (710 MB) runs over 8 parallel connections (for more: `CONNECTIONS=16 tools/install_oss_cad_suite.sh ~/opt`) and resumes where it stopped, both within a run and when you rerun the script. The finished file is checked against GitHub's SHA-256 for this release. Run it inside `tmux` or `screen` so a dropped SSH session doesn't stop it.
- If it keeps breaking, download the file on any machine with a better connection from <https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-09-28/oss-cad-suite-linux-x64-20260928.tgz>, copy it into `~/opt/` on the server (for example `scp oss-cad-suite-linux-x64-20260928.tgz SERVER:opt/`), and rerun the script. It checks the file's hash and installs from it.
- Add only `bin/` to PATH. Don't source the suite's `environment` script, which changes your Python setup.
- `yosys-slang` is the fuller SystemVerilog frontend that Phase 1 needs. This build already includes it.

**Check:** the last line reads `formal smoke test: OK`.

## Step 6: Check the teacher endpoint

`tools/teacher.py` talks to the teacher model through its OpenAI-compatible API, using only the Python standard library. Our teacher is GLM3-Flash, on a server you share with colleagues. Run the script from the machine that will run the data pipeline. Phase 0 only checks the connection and measures throughput; teacher labeling starts in Phase 1.

```bash
export TEACHER_BASE_URL=http://HOST:PORT/v1   # usually ends in /v1
export TEACHER_MODEL=MODEL_NAME               # `check` lists what the server serves
export TEACHER_API_KEY=KEY                    # only if the server needs one
python3 -I tools/teacher.py check
python3 -I tools/teacher.py bench --concurrency 4 --requests 16
```

- `check` lists the served models and sends one short request. If the model returns its reasoning separately, it says so.
- `bench` sends 16 requests that ask for SVA assertions, 4 at a time, and reports our output tokens/s and latency. That number sets the teacher budget for Phase 1.
- The endpoint is shared, so don't raise the concurrency to find its limit. Our data jobs will stay at about this level.

**Check:** both commands end in `[PASS]` lines. Note the output tokens/s from `bench`.

## Step 7: Build the benchmarks and the Phase 0 dataset

Three commands download both benchmarks, merge them, and turn them into training and test examples in the format of `spec/data_format.md`. Everything lands in `data/`, which git ignores.

```bash
tools/fetch_benchmarks.sh                       # ~10 s
python3 -I tools/build_benchmarks.py            # ~30 s; downloads ~1,000 RTL files from GitHub
python tools/build_phase0_data.py --tokenizer models/Qwen3-8B --max-seq-tokens 8192 --out data/phase0   # ~1-2 min
```

`fetch_benchmarks.sh` downloads hwe-bench from Hugging Face. If that hits the same proxy error, run it as `HF_ENDPOINT=https://hf-mirror.com tools/fetch_benchmarks.sh`; the file is pinned to the same revision either way.

Each hwe-bench bug fix yields three kinds of examples per changed file:

- **buggy:** a window around the bug, at a random offset. Files whose fix changes only comments or whitespace are skipped, and comment-only changes don't count as bug locations.
- **fixed:** the same window after the fix.
- **elsewhere:** a window of the buggy file away from the fix, weighted 0.5.

They all ask the same three questions: bug present, bug category, bug location. Each verification-benchmarks property becomes one "will formal checking find a violation?" question.

Your counts should match mine exactly. The pinned inputs are deterministic, and Qwen3-0.6B and 8B share a byte-identical tokenizer.

| File | Examples | Bug fixes / properties | Use |
| --- | --- | --- | --- |
| `train.jsonl` | 910 | 174 | training |
| `val.jsonl` | 113 | 16 | validation during training |
| `eval_hwe.jsonl` | 1,029 | 124 | hwe-bench test |
| `eval_vb.jsonl` | 119 | 119 | verification-benchmarks test |

**Check:** `data/phase0/stats.json` shows these counts, and `tokens_max` stays under 8,192.

## Step 8: GPU smoke run on 2 cards

A short end-to-end run on the small model, to prove the GPU path works before spending hours on 8B. In about 10 minutes it:

1. Trains 20 LoRA steps on Qwen3-0.6B across 2 GPUs (DDP over NCCL).
2. Validates at steps 10 and 20.
3. Saves `runs/smoke_gpu/final`.
4. Evaluates 64 examples from each test set.

```bash
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 launch/smoke_gpu.sh
```

The log prints lines like `step 10/20  loss 1.9  lr ...  tok/s ...  mem ...G`, then validation and test metrics. The accuracy numbers mean nothing yet: 20 steps on a 0.6B model only prove the pipeline runs. I ran this same script on CPU, with tiny settings, before handing it over.

**Check:** the last line reads `GPU smoke run OK`. Logs are in `runs/smoke_gpu/train.log` and `runs/smoke_gpu/eval.log`.

## Step 9: First real run: zero-shot baseline, then Qwen3-8B LoRA

First measure what Qwen3-8B does with no training, then train LoRA on the 910 Phase 0 examples and evaluate again on the same test sets. The comparison is the first real result.

```bash
# a) zero-shot baseline, about 10-15 min on 4 cards
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 launch/eval.sh
#    the same, but bug location is read from the block number Qwen would write (the untrained pointer head
#    can only guess); a few minutes longer, output in runs/zeroshot-Qwen3-8B-lmpointer/
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 POINTER=lm launch/eval.sh

# b) LoRA training (2 epochs, ~114 steps) plus evaluation, about 30 min on 4 cards
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 launch/train.sh
```

On 2 cards, use `CUDA_VISIBLE_DEVICES=0,1 NGPU=2` for both commands. The scripts keep 16 sequences per optimizer step, so expect about twice the time.

| Output | Where |
| --- | --- |
| Zero-shot metrics | `runs/zeroshot-Qwen3-8B/eval.log`, `*.metrics.json` |
| Training log | `runs/qwen3-8b-lora-v1/train.log` (loss, tok/s, peak memory, validation every 25 steps) |
| Checkpoints | `runs/qwen3-8b-lora-v1/step-50/`, `step-100/`, `final/` (LoRA adapter + pointer head, ~0.7 GB each) |
| Test metrics after training | `runs/qwen3-8b-lora-v1/eval.log`, `eval/*.metrics.json`, `eval/*.predictions.jsonl` |

What to expect:

- **Zero-shot:** `bug_location` is exactly chance, because the pointer head starts uniform; the `POINTER=lm` run reads it from Qwen's own answer instead. `bug_present` and `bug_category` show what Qwen3-8B knows untrained.
- **The extra lines:** under `bug_location`, `buggy` and `no bug` split the score. `buggy` = how often it points at a block the fix changed on files with a bug; `no bug` = how often it says "none" on the others. 58% of the test files have no bug, so answering "none" everywhere already scores 0.577 overall. `buggy, blocks only` ranks just the blocks on buggy files, so it measures finding the bug even when the model doubts there is one; chance is printed beside it. `pairs` (every hwe-bench question) is the share of buggy/fixed pairs where the buggy version gets the higher bug probability. 0.5 is chance; it is the most direct test of whether the model tells a bug from its fix.
- **After training:** calibration (ECE, NLL) improves just by learning how often each answer is right. Learning more shows as `pairs` above 0.5 and `blocks only` above chance. The first LoRA run (v0, Oct 14, in `runs/qwen3-8b-lora-v0/`) clearly got the calibration part. The defaults now follow recipe v1, which trains a bug's buggy and fixed versions in the same step; the plan's Phase 2 results explain why.
- **Detecting bugs in new projects:** don't expect a big jump in `bug_present` on the test projects. 174 OpenTitan bug fixes are too few for that; Phase 1's data is what fixes it.
- **Memory:** peak memory should stay far below the cards' capacity. The `mem` column in the log shows it.

**Check:** `train.log` shows the loss going down, and both `eval.log` files list metrics for all four questions.

## What to send back, and fixes for common failures

Paste these back to me when you're done, or at the first step that fails. For any error, paste the full traceback, not just its last line.

- [ ] Output of `python tools/check_env.py` (step 3)
- [ ] Output of `tools/formal_smoke.sh` (step 5)
- [ ] Output of `teacher.py check` and `teacher.py bench` (step 6)
- [ ] `data/phase0/stats.json` (step 7)
- [ ] `runs/smoke_gpu/train.log` and `runs/smoke_gpu/eval.log` (step 8)
- [ ] `runs/zeroshot-Qwen3-8B/eval.log`, `runs/qwen3-8b-lora-v1/eval.log`, and `grep -E "^step|val @" runs/qwen3-8b-lora-v1/train.log` (step 9)

| Symptom | Likely cause and fix |
| --- | --- |
| `torch.cuda.is_available()` is False | A CPU-only or mismatched wheel. Reinstall torch with the index URL from step 2 |
| `pip` can't find `torch==2.14.0` | Python is outside 3.10–3.13, or the index URL is wrong |
| `python: command not found`, or old packages, in a launch script | The venv isn't active. Run `source .venv/bin/activate`, or prefix the command with `PYTHON=.venv/bin/python` |
| NCCL error, or a hang when training starts | Another job is using the cards, or they lack a usable P2P path. Rerun with `NCCL_DEBUG=INFO` and send me the log |
| CUDA out of memory | Another process holds memory on the card (check `nvidia-smi`), or sequences are too long. Rebuild data with `--max-seq-tokens 6144` and pass `MAX_SEQ=6144` |
| `download_models.sh` fails, or `ProxyError: 504 Gateway Time-out` | The proxy can't reach the download site. Bypass the proxy or switch the source (step 4); re-running resumes |
| `CERTIFICATE_VERIFY_FAILED`, `self-signed certificate in certificate chain` | The proxy inspects HTTPS. Point Python at the system certificates (step 4) |
| `models/Qwen3-8B not found` | Step 4 didn't finish, or the command isn't running from the repo root |
| `build_benchmarks.py` reports download failures | A proxy timeout (`HTTP 504` under `hwe_download_errors`), a GitHub rate limit or a network blip. Rerun it: it downloads only the missing files. If they keep failing, add `--workers 4` |
| The OSS CAD Suite download is slow or keeps breaking | Rerun `install_oss_cad_suite.sh`; it resumes. Try `CONNECTIONS=16`, or copy the `.tgz` in from another machine (step 5) |
| `formal smoke test: FAILED` | Usually `PATH` lacks `oss-cad-suite/bin`. The script prints the path of `sby.log` |
| `teacher.py`: `cannot reach ...` | Wrong URL (it usually needs `/v1`), the server is down, or a firewall is in the way |

## Reference: scripts

Every script below was run on the dev box before handoff. The GPU scripts were run on CPU with tiny settings, so only NCCL and the GPU-speed checks are untested.

| Script | What it does | Step |
| --- | --- | --- |
| `tools/check_env.py` | Library versions, GPUs, bf16 matmul speed, NVLink topology, NCCL bandwidth, downloaded models, free disk | 3 |
| `tools/download_models.sh` | Downloads Qwen3-0.6B / 1.7B / 8B into `models/`, from ModelScope (or Hugging Face with `SOURCE=hf`) | 4 |
| `tools/install_oss_cad_suite.sh` | Installs the pinned OSS CAD Suite build without root | 5 |
| `tools/formal_smoke.sh` | SymbiYosys proves one property and catches one injected bug; checks yosys-slang | 5 |
| `tools/teacher.py` | Teacher client: `check` (connection) and `bench` (throughput); `--mock` for testing | 6 |
| `tools/fetch_benchmarks.sh` | Downloads verification-benchmarks and hwe-bench at pinned versions | 7 |
| `tools/build_benchmarks.py` | Merges both into `data/benchmarks/benchmarks.jsonl`, with buggy/fixed RTL and changed lines | 7 |
| `tools/build_phase0_data.py` | Builds `data/phase0/` training and test examples | 7 |
| `launch/smoke_gpu.sh` | Short end-to-end GPU run on Qwen3-0.6B | 8 |
| `launch/eval.sh` | Test-set evaluation: zero-shot, or a checkpoint via `CHECKPOINT=` | 9 |
| `launch/train.sh` | LoRA training on Qwen3-8B, then test-set evaluation | 9 |
| `launch/smoke_cpu.sh` | The CPU version of the whole pipeline (dev box only) | — |
| `jev/train.py`, `jev/eval.py` | Training and evaluation entry points (`python -m jev.train ...`, or under torchrun) | — |
| `jev/format.py`, `jev/model.py` | Prompt rendering with answer slots; LM-head and pointer readouts, LoRA, checkpoints | — |

Launch scripts take their settings from environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `NGPU` | Number of processes / GPUs | 2 (smoke), 4 (train, eval) |
| `MODEL` | Base model directory | `models/Qwen3-0.6B` (smoke), `models/Qwen3-8B` |
| `RUN` / `OUT` | Output directory | under `runs/` |
| `EPOCHS`, `LR`, `SEQ_PER_STEP` | Training length, LoRA learning rate, sequences per optimizer step | 2, 1e-4, 16 |
| `MAX_SEQ` | Token limit; must match the data build | 8192 |
| `CHECKPOINT` | Checkpoint for `eval.sh` (unset = zero-shot) | unset |
| `POINTER` | How `eval.sh` answers bug location: `head` (pointer head) or `lm` (the block number Qwen would write) | `head` |
| `EXTRA` | Extra `jev.train` flags, e.g. `EXTRA="--max-steps 10"` | empty |
| `PYTHON` | Python to use | `python` |
