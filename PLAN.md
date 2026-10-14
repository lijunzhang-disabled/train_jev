# Jev-like Hardware Security Model: Working Plan

As of Oct 14, 2026 · Lijun Zhang

> This is the git-tracked copy of the plan. The live version is the Claude Doc at
> https://claude.ai/code/artifact/b5e47478-1f63-41d8-80c5-6225dd0ea7d2. Claude updates both together.

## Goal and status

We're building a Jev-style decision model for chip/hardware security. It reads a hardware design once and answers many typed questions, each with a calibrated probability, without generating text. We start on general hardware data and specialize to security from Phase 5. This is a personal, non-commercial learning project, so datasets with research-only licenses are allowed.

Week 1 starts Oct 12, 2026. Change the Status column as phases move.

| Phase | Delivers | Weeks | Status |
| --- | --- | --- | --- |
| 0. Setup | Repo skeleton, hardware tools, GPU environment, smoke runs | 1 | In progress |
| 1. Dataset v1 | General hardware dataset with tool, by-construction and teacher labels | 1–2 | Not started |
| 2. Eval and baselines | Eval harness; zero-shot Qwen and DeepSeek numbers | 2 | Not started |
| 3. SFT v1 | Qwen3-8B LoRA model with calibrated probabilities | 3 | Not started |
| 4. Ablations | Fixed training recipe | 4 | Not started |
| 5. Security | Security questions, data and model, scored on real buggy designs | 5–8 | Not started |
| 6. Later | Serving, RL from real outcomes, distillation | after 8 | Not started |

## Settled decisions

We fine-tune a pretrained Qwen and train it with log loss against labels we trust. That reproduces Jev's "calibrated decisions" (RLCD) goal without RL until real outcome feedback exists.

| Area | Choice | Why |
| --- | --- | --- |
| Backbone | Qwen3-8B; debug on Qwen3-1.7B. Try a newer base model such as Qwen3.5-9B only after the pipeline works end to end. Qwen3.6 and 3.8 only come in 27B and larger | Plain transformer with mature tooling; Qwen3.5-9B mixes in linear-attention layers that need extra kernels; 1.7B runs ~5x faster for debugging |
| Inference | One forward pass. Each question gets an `<ans>` slot; logits only over that question's options (LM head sliced to those rows) | No generation; many questions per pass; no full-vocabulary logits |
| Objective | Log loss on soft labels, weighted by label source; temperature scaling afterwards | Log loss is a proper scoring rule, so probabilities are calibrated when the labels are true outcomes |
| Label trust | Tool-verified > by construction > teacher > human gold set (eval only) | Calibrate to facts, not to DeepSeek's opinions |
| Teacher | DeepSeek V4.1 Flash writes scenarios and bug injections, and labels semantic questions with 3–5 samples each | Runs on its own host; this box can't reach it |
| RL | Not before Phase 6 | Only needed once real outcomes arrive with selection bias |
| Parallelism | LoRA + DDP first; FSDP full fine-tuning on 4 cards if LoRA plateaus | LoRA fits on one card; full fine-tuning needs ~128 GB of training state |
| Serving | One model copy per GPU, forward pass only | 8B in bf16 is 16 GB; no decoding, so tensor parallelism doesn't help |
| Compute | 2–4 H20 cards; you set up and launch all jobs on the server | No SSH from this box |

## Task and data

We copy the shape of [verification-benchmarks](https://github.com/HWSec-UNC/verification-benchmarks): an RTL design, an SVA property, a CWE tag and a formal-verification result. Its 119 labeled properties become our security test set. We generate training data in the same shape and label it with open-source formal tools.

**What the benchmark contains.** Each property comes with the bug it targets, the module, a CWE ID, and a result from Cadence JasperGold FPV, a commercial tool. 87 of the 119 results are "Violation found" and 32 are "No violation found".

| Design folder | Base design | Known bugs | Properties written |
| --- | --- | --- | --- |
| [hackatdac18](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/hackatdac18/README.md) | PULPissimo SoC, RI5CY core | 31 | 20 |
| [hackatdac19](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/hackatdac19/README.md) | Ariane (CVA6) SoC | 66 | 11 |
| [hackatdac21](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/hackatdac21/README.md) | OpenPiton + Ariane SoC | 99 | 17 |
| [or1200](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/or1200/README.md) | OR1200 CPU with bugs from prior work | — | 71, each also tagged with a category (6 labels in the table) |

**Candidate question types** (you're still choosing; A is my recommendation for the core task):

| Type | Example question | Answer type | Label from | Phase |
| --- | --- | --- | --- | --- |
| A. Property outcome | Will formal checking find a violation of this assertion? | bool | SymbiYosys check or proof | 1 general, 5 security |
| B. Bug presence | Does this module contain a functional bug? | bool | Injected bug + equivalence check | 1 |
| C. Bug class | Which kind of bug is present? | enum: reset value, condition, address range, FSM, access check, data not cleared, none | Injected bug | 1, 5 |
| D. Bug location | Which always block (or signal) is wrong? | pointer head: a probability for each block marker in the input | Injected bug | 1 |
| E. CWE | Which hardware CWE fits? | enum over the 26 CWEs used in the benchmark + none | Injection template; teacher | 5 |
| F. OR1200 category | Control flow, privilege change, register update, memory access, or computation? | enum | Injection template; teacher | 5 |
| G. Property–bug match | Does this assertion detect the described bug? | bool | Check on buggy vs clean design | 5 |
| H. Structure | Has an FSM? Reset style? Multiple clock domains? Bus protocol? Module function? | bool / enum | Tools; teacher | 1 |

Why A as the core: it's exactly the benchmark's label, and open tools can check it. It also gives the model a clear job: a fast first pass that ranks which properties are likely to fail before slow formal runs (System One before System Two).

One training item, as a JSONL line (pretty-printed):

```json
{
  "id": "cfg_lock-mut0042",
  "state": {
    "rtl": "module cfg_lock (input clk, rst_n, we, input [31:0] wdata, ...); ... endmodule",
    "context": "Config register; lock_q should block writes once set."
  },
  "questions": [
    {"id": "q1", "type": "bool",
     "text": "Will formal checking find a violation of: assert property (@(posedge clk) lock_q && we |=> cfg_q == $past(cfg_q));"},
    {"id": "q2", "type": "enum", "text": "Which bug class is present?",
     "options": ["none", "reset value", "inverted condition", "missing access check", "FSM transition"]}
  ],
  "labels": {
    "q1": {"dist": {"yes": 1.0}, "source": "tool:sby-prove"},
    "q2": {"dist": {"missing access check": 1.0}, "source": "construction"}
  }
}
```

We have no JasperGold. Our own labels come from SymbiYosys with the yosys-slang SystemVerilog frontend, run module by module. Heavy SystemVerilog may not parse, so modules that fail are dropped.

## Data sources

We have two real-bug test sets: verification-benchmarks for security and hwe-bench for general RTL bugs. The Hugging Face datasets below supply the training corpus, ready-made properties, and pairs of correct and buggy designs. Except for hwe-bench, I've only checked their Hugging Face metadata so far, not their files.

| Source | What it gives us | Size | License | Use |
| --- | --- | --- | --- | --- |
| [verification-benchmarks](https://github.com/HWSec-UNC/verification-benchmarks) | SVA security properties on buggy SoCs, each with a CWE and a JasperGold result | 119 properties | Not stated | Security test set (Phase 5) |
| [hwe-bench](https://github.com/pku-liang/hwe-bench) | Real RTL bug fixes: problem statement, buggy commit, fix patch, and a test that fails before the fix and passes after | 417 cases; 321 are SystemVerilog RTL fixes | Apache-2.0 | General test set for types B–D; older OpenTitan cases are the Phase 0 training data |
| [wyt2000/CodeV-SVA-datasets](https://huggingface.co/datasets/wyt2000/CodeV-SVA-datasets) | 42K RTL modules; 324K RTL + spec + SVA triples whose SVAs were never checked against the RTL | 324K triples | Apache-2.0 | Main type A source: SymbiYosys labels each SVA as violated or holding |
| [AnonymousHierSVA/HierSVA](https://huggingface.co/datasets/AnonymousHierSVA/HierSVA) | BaseJump STL modules with formally proven SVA, plus 28 buggy variants | 342 modules | Solderpad SHL-2.1 | Best-quality A and G labels; multi-clock seeds |
| [aditigupta/EquivSVA](https://huggingface.co/datasets/aditigupta/EquivSVA) | 120 behavior families × 4 equivalent RTL variants, gold properties, 3 typed mutants each | 480 RTL | CC BY 4.0 | A, G, B, C; split by family |
| [architect-ubc-capstone/rtl-augmented-v3](https://huggingface.co/datasets/architect-ubc-capstone/rtl-augmented-v3) | Injected bugs in 49 GitHub repos, with diffs and bug types | 948 bugs | MIT (upstream varies) | B, C, D; strip comments first, because some leak the label |
| [scale-lab/MetRex](https://huggingface.co/datasets/scale-lab/MetRex) | Synthesizable RTL modules | 25,868 | BSD-3-Clause | Clean pool for mutants and properties |
| [theepicflyer/openrtlset-permissive-verified](https://huggingface.co/datasets/theepicflyer/openrtlset-permissive-verified) | Lint-clean RTL with repo names | 13,625 | CC BY 4.0 | Clean pool with exact repo filtering |
| [HPAI-BSC/SystemVerilog-SFT-86k](https://huggingface.co/datasets/HPAI-BSC/SystemVerilog-SFT-86k) | Test plans and testbenches generated from RTL | 85,701 | Not stated | Type H context; contains OpenTitan code, so filter first |

hwe-bench's 86 Chisel cases (XiangShan, rocket-chip) are out of scope for v1, because our pipeline reads Verilog and SystemVerilog. We only need its JSONL files and git checkouts of the projects, not its 200 GB of Docker images.

`tools/build_benchmarks.py` merges both benchmarks into `data/benchmarks/benchmarks.jsonl`: 119 verification-benchmarks records, all test, and 321 hwe-bench records, 126 test and 195 train under the proposed split. Each hwe-bench record links the buggy RTL files, the files with the fix applied, and the changed line ranges, which are the bug-location labels. All 1,000 patched files match GNU `patch` output. `test_exclusions.txt` lists 532 files that training data must not include.

Demoted after inspection: shailja/Verilog_GitHub (not actually deduplicated, and it contains OR1200 code), HPAI-BSC/SystemVerilog-RL-12k (its mutants are gate-level netlists), and TroLL (gate-level Trojans whose net names leak the label). The full survey, including datasets built from the test families that must never be trained on, is in `docs/hf_dataset_survey.md`.

**Phase 0 training data is small.** It comes from 174 OpenTitan bug fixes: 910 examples, about 4.0M tokens at an 8k window. That's enough to prove the pipeline end to end, but it's about 2% of Phase 1's ~40k-item target, all from one project. Phase 1 scales it with the sources above, more mined OpenTitan fixes (2,594 merged "fix" PRs before 2024), and our own mutants. Real files are also long: 27% of the hwe-bench training files exceed 8k tokens, and the largest is 182k. So each example is a window placed at a random offset, which keeps the bug's position in the window from giving away its location.

## Workflow

I write and test all code on the dev box. You copy it to the server, run it, and paste back logs. The data pipeline has to run somewhere that can reach DeepSeek, because the dev box can't.

- **Claude:** code, configs, launch scripts, CPU smoke tests (Qwen3-0.6B plus a mock teacher), debugging from logs, keeping this doc current.
- **You:** server setup, the DeepSeek endpoint, running data and training jobs, labeling the human gold sets, pasting back logs and metrics.

```
+---------------------------+                +---------------------------+              +---------------------------+
| Dev box (here)            |  -- code -->   | H20 server                | -- prompts ->| DeepSeek host             |
| No GPU, no DeepSeek       |                | You set up and run jobs   |              | DeepSeek V4.1 Flash       |
| Claude writes all code    |                | Data pipeline + tools     |              | Writes properties and     |
| CPU smoke tests with      |  <- - - - - -  | (SymbiYosys, Verilator)   | <- outputs --| bug injections            |
| Qwen3-0.6B + mock teacher |   your logs    | Training on 2-4 H20s      |              | Labels semantic questions |
|                           |                | Eval and metrics          |              |                           |
+---------------------------+                +---------------------------+              +---------------------------+
```

Code reaches the server by git or a manual copy, and logs come back only when you paste them. The link to DeepSeek assumes the server side can reach it, which is still an open question.

## Phase 0: Setup (week 1)

Phase 0 ends when one smoke run passes on CPU here and on 2 GPUs there, SymbiYosys proves a toy property, and DeepSeek answers through our script.

**Claude, on the dev box**

- [x] `git init`, with the GitHub remote
- [x] Fetch both benchmarks and convert them into one record format: `tools/fetch_benchmarks.sh`, `tools/build_benchmarks.py`
- [x] Repo skeleton: `spec/`, `jev/`, `launch/`, `docs/`
- [x] Data format spec v0 (`spec/data_format.md`) and a mock mode for the teacher client
- [x] Readout and training loop; CPU smoke test with Qwen3-0.6B (one process and 2-process DDP)
- [x] Launch scripts for 2 and 4 GPUs, plus a GPU smoke test for you to run
- [x] Step-by-step server guide: [Phase 0 Setup Guide](https://claude.ai/code/artifact/95cbaea8-5860-4ab6-b94c-7f0dddbb66aa), also `docs/phase0_guide.md`

**You, on the server side:** follow the [Phase 0 Setup Guide](https://claude.ai/code/artifact/95cbaea8-5860-4ab6-b94c-7f0dddbb66aa) step by step.

- [x] PyTorch, transformers and peft at the versions in requirements.txt (flash-attn isn't needed)
- [ ] `nvidia-smi topo -m`: the cards you use should show `NV18` between them
- [x] Download Qwen3-0.6B, Qwen3-1.7B and Qwen3-8B from ModelScope into `models/`
- [ ] Install OSS CAD Suite (Yosys, SymbiYosys, solvers, Verilator, Icarus) on the machine that will run the data pipeline; check that the yosys-slang plugin loads
- [ ] Confirm the DeepSeek endpoint is OpenAI-compatible and reachable from that machine; measure tokens/s at batch 32
- [x] Run `tools/fetch_benchmarks.sh`, then `python3 -I tools/build_benchmarks.py` (under a minute; downloads about 1,000 RTL files from GitHub)
- [ ] Run the GPU smoke test and paste back the log

## Phase 1: General hardware dataset v1 (weeks 1–2)

The target is about 40k RTL modules with 5–10 questions each, labeled mostly by tools. No security questions yet: question types A–D and H.

1. **Collect the corpus.** Take RTL from outside the test families: the clean module pools MetRex and openrtlset-permissive-verified, projects such as PicoRV32 and BaseJump STL, and OpenTitan files not listed in test_exclusions.txt. Split it into modules. Keep only modules that parse with yosys-slang.
    - Leave out the benchmark's design families (PULPissimo/RI5CY, Ariane/CVA6, OpenPiton, OR1200), plus every file an hwe-bench test case touches, so both test sets measure generalization.
2. **Remove duplicates and split.** Remove near-duplicates with MinHash. Split train/val/test by project family, never by file.
3. **Inject bugs** (types B, C, D). Use rule-based mutations (reset value, inverted condition, off-by-one, FSM transition, swapped signals, removed check) plus realistic bugs written by DeepSeek. Run an equivalence check on each mutant: one that behaves the same as the original is labeled "no bug". Older hwe-bench OpenTitan cases, plus bug fixes mined from project history with hwe-bench's scripts, add real bugs.
4. **Write properties** (type A). Start from CodeV-SVA's 324K RTL + SVA pairs, which only need SymbiYosys labels; DeepSeek writes assertions for modules that have none. SymbiYosys checks each one on the clean version and on each mutant.
    - A proof or a counterexample becomes a hard label. "No violation within k cycles" gets a softer label, and timeouts are dropped.
    - Aim for roughly half violated and half holding.
5. **Answer structural questions** (type H) with tools where possible. DeepSeek labels the rest with 3–5 samples each.
6. **Check the teacher.** On type A and B questions, compare DeepSeek's answers with the tool labels. Its accuracy there sets the weight its labels get elsewhere.

Phase 1 ends with dataset v1, its statistics (items per type, label balance, label sources) and the teacher-vs-tool agreement report.

## Phase 2: Eval harness and baselines (week 2)

Phase 2 ends with a table of baseline numbers that every later model must beat.

**Test sets, none used for training**

- Held-out project families from the Phase 1 split
- Security: 119 verification-benchmarks properties, plus the Hack@DAC bug descriptions. General: hwe-bench's test-only cases (see open questions)
- A human gold set of about 300 items that you label

**Metrics**

- Accuracy and macro-F1 per question type
- For type A: AUROC, and recall at a fixed false-alarm rate
- Calibration: ECE, Brier score, reliability diagrams
- 73% of verification-benchmarks results are violations (87 of 119), so report AUROC and calibration next to accuracy

**Baselines**

- Majority class
- Zero-shot Qwen3-8B with the constrained readout
- DeepSeek on a sample of the same questions; this sets a teacher ceiling and shows where the teacher is wrong

**Results so far.** Zero-shot Qwen3-8B (Oct 14, 2026, 4 H20s, cleaned test set) is no better than chance at telling buggy code from fixed code, or violated properties from holding ones, and it can't find bugs either. It is also overconfident: its log loss is 1.5–5× that of always predicting the label frequencies. The model has to beat both the base rate and these numbers.

| Question | Test set | n | Base-rate acc | Base-rate NLL | Qwen3-8B acc | Macro-F1 | AUROC | NLL | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bug_present (yes/no) | hwe-bench test | 1,029 | 0.577 | 0.681 | 0.526 | 0.493 | 0.484 | 2.837 | 0.402 |
| bug_category (6 options) | hwe-bench test | 1,029 | 0.577 | 1.292 | 0.535 | 0.169 | | 1.973 | 0.240 |
| bug_location (about 25 blocks + none), pointer head | hwe-bench test | 1,029 | 0.577 | 1.980 | 0.101 | | | 3.055 | 0.036 |
| bug_location, read from Qwen's written answer | hwe-bench test | 1,029 | 0.577 | 1.980 | 0.577 | | | 6.276 | 0.421 |
| property_violated (yes/no) | verification-benchmarks | 119 | 0.731 | 0.582 | 0.286 | 0.237 | 0.449 | 2.820 | 0.662 |

Base rate = always answering the most common label (acc) or predicting the label frequencies (NLL). For bug_location the most common answer is "none" (594 of the 1,029 files have no bug), and the NLL baseline says "none" with that probability and spreads the rest evenly over the blocks. Pointing at any block the fix changes counts as correct, since 61% of the bugs span several blocks.

- **Pointer head:** untrained, so exactly uniform: top-3 0.262; on the 435 buggy files, acc 0.140 and top-3 0.341.
- **Qwen's written answer** (`POINTER=lm`, the block number or "none" Qwen would write after "Q3:"): the 0.577 is the share of files without a bug, so Qwen almost certainly answers "none" on nearly every file. Its top-3 is 0.723 against 0.689 for the baseline. Its NLL is 3× the baseline's because it is confidently wrong on the buggy files. Its Q1 answer meanwhile says "yes" on roughly a third of the files, so its answers contradict each other. The eval log now prints a buggy/no-bug split under bug_location; `python -m jev.metrics runs/zeroshot-Qwen3-8B-lmpointer/eval_hwe.predictions.jsonl` prints it from the saved predictions.
- **Rerun noise:** the two runs' other numbers differ in the third decimal (one or two near-ties flipping in bf16 on the GPU).
- **First build:** on the 1,045-example test set the numbers were within 0.01 of these.

## Phase 3: SFT v1 (week 3)

An 8B LoRA run on dataset v1 takes about 9 hours on 4 cards. Phase 3 ends when v1 clearly beats zero-shot on held-out families, ECE after temperature scaling is ≤ 0.05, and type-A AUROC on the benchmark is reported.

1. Debug the whole pipeline on Qwen3-1.7B.
2. Train Qwen3-8B with LoRA (rank 64, all linear layers). The loss is KL to the soft labels at the `<ans>` slots only, weighted by label source.
3. Fit temperature scaling on the validation set.

Estimated run times for ~40k items × ~3k tokens × 2 epochs ≈ 240M tokens, assuming ~40% GPU utilization:

| Run | 4×H20 (hours) | 2×H20 (hours) |
| --- | --- | --- |
| Qwen3-1.7B LoRA | ~2 | ~4 |
| Qwen3-8B LoRA | ~9 | ~18 |
| Qwen3-8B full fine-tune (FSDP) | ~14 | ~28 |

## Phase 4: Ablations (week 4)

Phase 4 fixes the training recipe; record each result in the decision log below.

- LoRA vs full FSDP fine-tuning on 4 cards
- Label-source weights, and the soft-label value for "no violation within k cycles"
- Context length (4k vs 8k), and how much of the surrounding module hierarchy to include
- A newer base model (Qwen3.5-9B) vs Qwen3-8B, once the pipeline works end to end
- Training with vs without the benchmark's design families, to measure how much familiarity with a design helps

## Phase 5: Chip security (weeks 5–8)

Phase 5 ends with a target recall at a fixed false-alarm rate on the benchmark, with calibrated probabilities. It adds question types A (security properties), C, E, F and G, plus "escalate to security review".

1. **CWE label set.** Use the 26 CWE IDs the benchmark already uses (e.g., CWE-1245, CWE-1260, CWE-1234, CWE-226), plus "none".
2. **Security bug templates.** Base them on the bug patterns in the benchmark:
    - Lock register writable after lock, or cleared on reset
    - Debug access without authentication, or an incomplete password check
    - Key or plaintext not cleared after use or on entering debug mode
    - Overlapping protected address ranges
    - FSM hang or unreachable state
    - Wrong privilege level, or a CSR accessible from a lower privilege
3. **Labels by construction.** DeepSeek injects a template into a clean training module and writes the SVA property that should catch it. SymbiYosys confirms that the property fails on the injected version and holds on the clean one. That gives labels for types A and G, with the CWE and category from the template.
4. **Negatives and base rate.** Pair each injected module with its clean version and benign look-alikes. Keep a realistic base rate: most modules are clean.
5. **Train.** Continue from the Phase 4 checkpoint, mixing in general hardware data so the model keeps what it learned.
6. **Evaluate** on the 119 benchmark properties, the Hack@DAC bug descriptions, and a security gold set you review.

## Phase 6: Later (after week 8)

Once the security model works, Phase 6 closes the loop between fast predictions and slow formal checks.

- **Serving:** one model copy per GPU, forward pass only, behind a small API.
- **Predict, then verify:** the model ranks properties by how likely they are to fail; SymbiYosys checks the top ones; the results come back as new labels.
- **RL-style calibration from real outcomes:** formal checks only run on what the model flagged, so the new labels are biased toward flagged items. Correct for that with propensity weighting.
- **Distillation** into a smaller model, if speed matters.

## Risks

The biggest risk is open-source formal tools failing on heavy SystemVerilog, because type A labels depend on them.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Yosys can't parse heavy SystemVerilog | Lost modules; no labels | yosys-slang frontend; check module by module; drop failures; fall back to Verilator simulation, which can show violations but not prove properties |
| Bounded checks aren't proofs | "Holds" labels may be wrong | Prove mode (k-induction) where possible; softer labels for bounded-only results |
| Mutants that change nothing | False "bug" labels | Equivalence check on every mutant |
| Duplicates or design families shared by train and test | Inflated scores | MinHash dedup; split by family; keep benchmark families and hwe-bench test files out of training |
| Security benchmark is small (119) and 73% violations | Noisy eval | Report recall and calibration with confidence intervals; add our own held-out security set |
| DeepSeek is weak on hardware meaning | Bad soft labels | Measure it against tool labels in Phase 1; lower its weight |
| Synthetic bugs don't look like real ones | Poor transfer | Test on Hack@DAC bugs; prefer realistic injections written by DeepSeek over rule-based mutations |
| RTL longer than the context window | Truncated states | Windows at random offsets, so the bug's position carries no signal (27% of hwe-bench files exceed 8k tokens); module-level states for synthetic data |
| Slow DeepSeek throughput | Data generation stalls | Measure in Phase 0; do the tool-labeled types first |

## Open questions and decision log

Five questions are open. The most important is which question types make up v1.

- [ ] Which question types go into v1? (Proposal: A as the core, plus B, C, D and H)
- [ ] Which hwe-bench cases are test-only? (Proposal: all Ibex, CVA6 and Caliptra cases, plus OpenTitan cases from 2024 on, for 126 in total. The 195 older OpenTitan cases become real-bug training data.)
- [ ] Can the H20 server, or a machine next to it, reach DeepSeek? That's where the data pipeline will run.
- [ ] Are the H20 cards 96 GB or 141 GB?
- [ ] Do you have JasperGold access? The plan assumes open-source tools only.

| Date | Decision |
| --- | --- |
| Oct 14, 2026 | Zero-shot baseline rerun on the cleaned test set (1,029), plus bug location read from Qwen's written answer: it says "none" almost everywhere. The bug_location baseline is now "always none" (acc 0.577, NLL 1.980) instead of a uniform guess, and the eval prints the buggy/no-bug split; `python -m jev.metrics` re-scores saved predictions |
| Oct 14, 2026 | Phase 0 data skips files whose fix changes only comments or whitespace, and comment-only changes no longer count as bug locations. An i2c comment typo fix had been labeled a logic error. Test set: 1,045 → 1,029 examples; training: 933 → 910 |
| Oct 14, 2026 | Zero-shot Qwen3-8B baseline recorded under Phase 2. Accuracy and top-3 give tied options equal credit, so a uniform guess scores chance, and any block the fix changes counts as the right location |
| Oct 13, 2026 | Models come from ModelScope into `models/`, because the server's proxy times out on Hugging Face. Its Qwen3 files are byte-identical to Hugging Face's |
| Oct 12, 2026 | Phase 0 code written and tested on CPU; the server steps are in the Phase 0 Setup Guide |
| Oct 12, 2026 | flash-attn dropped; PyTorch's SDPA attention is enough on H20 |
| Oct 12, 2026 | Hugging Face sources chosen from the survey: CodeV-SVA, HierSVA and EquivSVA for A and G; rtl-augmented-v3 for B–D; MetRex and openrtlset-permissive-verified as clean pools |
| Oct 12, 2026 | hwe-bench added as the general real-bug test set; its Chisel cases are out of scope for v1 |
| Oct 12, 2026 | Bug location (type D) uses a pointer head over block markers |
| Oct 12, 2026 | Qwen3-8B until the pipeline works end to end; newer base models come after that |
| Oct 12, 2026 | Task shape follows verification-benchmarks; its 119 properties are test-only |
| Oct 12, 2026 | Personal, non-commercial project; datasets with research-only licenses allowed |
| Oct 12, 2026 | The dev box can't reach DeepSeek; the data pipeline runs on the server side |
| Oct 12, 2026 | Domain is chip/hardware security, starting with general hardware data |
| Oct 12, 2026 | No RL before Phase 6; log loss on trusted labels instead |
| Oct 12, 2026 | Post-training only: Qwen3-8B, LoRA first, on 2–4 H20 cards |

## Sources

- [HWSec-UNC/verification-benchmarks](https://github.com/HWSec-UNC/verification-benchmarks): README and per-design READMEs and property tables
- [Security Properties for Open-Source Hardware Designs](https://arxiv.org/abs/2412.08769): the preprint the repo asks users to cite (SecDev 2025)
- [pku-liang/hwe-bench](https://github.com/pku-liang/hwe-bench): README; case counts computed from [henryen/hwe-bench](https://huggingface.co/datasets/henryen/hwe-bench) `hwe_bench_full.jsonl`
- [Qwen on Hugging Face](https://huggingface.co/Qwen): model list, plus the config files of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- Hugging Face dataset pages linked in Data sources: metadata only
