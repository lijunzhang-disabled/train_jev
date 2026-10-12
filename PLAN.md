# Jev-like Hardware Security Model: Working Plan

As of Oct 12, 2026 · Lijun Zhang

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
| Backbone | Qwen3-8B; debug on Qwen3-1.7B; compare against Qwen3.5-9B once in Phase 4 | Standard architecture, mature tooling; 1.7B runs ~5x faster for debugging |
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

**What the benchmark contains.** Each property comes with the bug it targets, the module, a CWE ID, and a result from Cadence JasperGold FPV, a commercial tool. Almost every result is "Violation found", because the properties were written to catch known bugs.

| Design folder | Base design | Known bugs | Properties written |
| --- | --- | --- | --- |
| [hackatdac18](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/hackatdac18/README.md) | PULPissimo SoC, RI5CY core | 31 | 20 |
| [hackatdac19](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/hackatdac19/README.md) | Ariane (CVA6) SoC | 66 | 11 |
| [hackatdac21](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/hackatdac21/README.md) | OpenPiton + Ariane SoC | 99 | 17 |
| [or1200](https://raw.githubusercontent.com/HWSec-UNC/verification-benchmarks/main/or1200/README.md) | OR1200 CPU with bugs from prior work | — | 71, each also in one of 5 categories |

**Candidate question types** (you're still choosing; A is my recommendation for the core task):

| Type | Example question | Answer type | Label from | Phase |
| --- | --- | --- | --- | --- |
| A. Property outcome | Will formal checking find a violation of this assertion? | bool | SymbiYosys check or proof | 1 general, 5 security |
| B. Bug presence | Does this module contain a functional bug? | bool | Injected bug + equivalence check | 1 |
| C. Bug class | Which kind of bug is present? | enum: reset value, condition, address range, FSM, access check, data not cleared, none | Injected bug | 1, 5 |
| D. Bug location | Which always block (or signal) is wrong? | enum over candidates listed in the question | Injected bug | 1 |
| E. CWE | Which hardware CWE fits? | enum over ~25 CWEs used in the benchmark + none | Injection template; teacher | 5 |
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

- [ ] `git init` and the repo skeleton: `spec/`, `tools/`, `jev/`, `launch/`, with `data/` ignored by git
- [ ] Data format spec v0 and a mock teacher, so the pipeline can be tested without DeepSeek
- [ ] Readout and training loop; a CPU smoke test with Qwen3-0.6B
- [ ] Launch scripts for 2 and 4 GPUs, plus a GPU smoke test for you to run

**You, on the server side**

- [ ] GPU driver and CUDA 12.x, PyTorch, transformers, peft, flash-attn
- [ ] `nvidia-smi topo -m`: the cards you use should show `NV18` between them
- [ ] Download Qwen3-0.6B, Qwen3-1.7B and Qwen3-8B from Hugging Face
- [ ] Install OSS CAD Suite (Yosys, SymbiYosys, solvers, Verilator, Icarus) on the machine that will run the data pipeline; check that the yosys-slang plugin loads
- [ ] Confirm the DeepSeek endpoint is OpenAI-compatible and reachable from that machine; measure tokens/s at batch 32
- [ ] `git clone --recursive` verification-benchmarks, plus the HACK-EVENT hackatdac18/19/21 repos
- [ ] Run the GPU smoke test and paste back the log

## Phase 1: General hardware dataset v1 (weeks 1–2)

The target is about 40k RTL modules with 5–10 questions each, labeled mostly by tools. No security questions yet: question types A–D and H.

1. **Collect the corpus.** Take open RTL projects (OpenTitan, Ibex, PicoRV32, and others) and split them into modules. Keep only modules that parse with yosys-slang.
    - Leave out the benchmark's design families (PULPissimo/RI5CY, Ariane/CVA6, OpenPiton, OR1200), so the security test set measures generalization.
2. **Remove duplicates and split.** Remove near-duplicates with MinHash. Split train/val/test by project family, never by file.
3. **Inject bugs** (types B, C, D). Use rule-based mutations (reset value, inverted condition, off-by-one, FSM transition, swapped signals, removed check) plus realistic bugs written by DeepSeek. Run an equivalence check on each mutant: one that behaves the same as the original is labeled "no bug".
4. **Write properties** (type A). DeepSeek writes SVA assertions for each clean module. SymbiYosys checks each one on the clean version and on each mutant.
    - A proof or a counterexample becomes a hard label. "No violation within k cycles" gets a softer label, and timeouts are dropped.
    - Aim for roughly half violated and half holding.
5. **Answer structural questions** (type H) with tools where possible. DeepSeek labels the rest with 3–5 samples each.
6. **Check the teacher.** On type A and B questions, compare DeepSeek's answers with the tool labels. Its accuracy there sets the weight its labels get elsewhere.

Phase 1 ends with dataset v1, its statistics (items per type, label balance, label sources) and the teacher-vs-tool agreement report.

## Phase 2: Eval harness and baselines (week 2)

Phase 2 ends with a table of baseline numbers that every later model must beat.

**Test sets, none used for training**

- Held-out project families from the Phase 1 split
- The benchmark: 119 verification-benchmarks properties, plus the Hack@DAC bug descriptions
- A human gold set of about 300 items that you label

**Metrics**

- Accuracy and macro-F1 per question type
- For type A: AUROC, and recall at a fixed false-alarm rate
- Calibration: ECE, Brier score, reliability diagrams
- The benchmark is almost all "Violation found", so score it on recall and calibration, not accuracy

**Baselines**

- Majority class
- Zero-shot Qwen3-8B with the constrained readout
- DeepSeek on a sample of the same questions; this sets a teacher ceiling and shows where the teacher is wrong

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
- Qwen3.5-9B vs Qwen3-8B
- Training with vs without the benchmark's design families, to measure how much familiarity with a design helps

## Phase 5: Chip security (weeks 5–8)

Phase 5 ends with a target recall at a fixed false-alarm rate on the benchmark, with calibrated probabilities. It adds question types A (security properties), C, E, F and G, plus "escalate to security review".

1. **CWE label set.** Use the ~25 CWE IDs the benchmark already uses (e.g., CWE-1245, CWE-1260, CWE-1234, CWE-226), plus "none".
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
| Duplicates or design families shared by train and test | Inflated scores | MinHash dedup; split by family; keep benchmark families out of training |
| Benchmark is small (119) and skewed toward violations | Noisy eval | Report recall and calibration with confidence intervals; add our own held-out security set |
| DeepSeek is weak on hardware meaning | Bad soft labels | Measure it against tool labels in Phase 1; lower its weight |
| Synthetic bugs don't look like real ones | Poor transfer | Test on Hack@DAC bugs; prefer realistic injections written by DeepSeek over rule-based mutations |
| RTL longer than the context window | Truncated states | Module-level states; 8k context; a short summary of the surrounding hierarchy |
| Slow DeepSeek throughput | Data generation stalls | Measure in Phase 0; do the tool-labeled types first |

## Open questions and decision log

Four questions are open. The most important is which question types make up v1.

- [ ] Which question types go into v1? (Proposal: A as the core, plus B, C, D and H)
- [ ] Can the H20 server, or a machine next to it, reach DeepSeek? That's where the data pipeline will run.
- [ ] Are the H20 cards 96 GB or 141 GB?
- [ ] Do you have JasperGold access? The plan assumes open-source tools only.

| Date | Decision |
| --- | --- |
| Oct 12, 2026 | Task shape follows verification-benchmarks; its 119 properties are test-only |
| Oct 12, 2026 | Personal, non-commercial project; datasets with research-only licenses allowed |
| Oct 12, 2026 | The dev box can't reach DeepSeek; the data pipeline runs on the server side |
| Oct 12, 2026 | Domain is chip/hardware security, starting with general hardware data |
| Oct 12, 2026 | No RL before Phase 6; log loss on trusted labels instead |
| Oct 12, 2026 | Post-training only: Qwen3-8B, LoRA first, on 2–4 H20 cards |

## Sources

- [HWSec-UNC/verification-benchmarks](https://github.com/HWSec-UNC/verification-benchmarks): README and per-design READMEs and property tables
- [Security Properties for Open-Source Hardware Designs](https://arxiv.org/abs/2412.08769): the preprint the repo asks users to cite (SecDev 2025)
