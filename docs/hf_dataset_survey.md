# Hugging Face dataset survey

Surveyed 2026-10-12 through the Hub API, the dataset-viewer API and sampled rows (no full downloads).
"Seen" means rows or files were inspected; "card" means the claim comes only from the dataset card.
Leakage = sampled rows that contain code from a test-set family (OpenTitan, Ibex, CVA6/Ariane,
PULPissimo/RI5CY/cv32e40p, OpenPiton, OR1200, Caliptra, and rocket-chip/XiangShan, which are also in
hwe-bench). A clean sample does not prove a dataset is clean, so filter before training anyway.

Question types: A property outcome, B bug present, C bug class, D bug location, E hardware CWE,
F bug category, G "does this assertion catch this bug?", H structure and semantics, T hardware Trojan.

## Ranked summary

| # | Dataset | Rows | License | Feeds | Leakage | Verdict |
|---|---|---|---|---|---|---|
| 1 | [wyt2000/CodeV-SVA-datasets](https://huggingface.co/datasets/wyt2000/CodeV-SVA-datasets) | 42K RTL; 324K RTL+spec+SVA (unfiltered); 83K refined | apache-2.0 | **A**, G, H | Unlikely (0/2000) | **Use now**: label the 324K SVAs with SymbiYosys |
| 2 | [AnonymousHierSVA/HierSVA](https://huggingface.co/datasets/AnonymousHierSVA/HierSVA) | 342 modules with proven SVA; 28 buggy variants | Solderpad SHL-2.1 | **A**, **G**, B/C/D (small), H incl. multi-clock | Unlikely (BaseJump STL) | **Use now**: best labels, small |
| 3 | [aditigupta/EquivSVA](https://huggingface.co/datasets/aditigupta/EquivSVA) | 480 RTL (288/96/96 by family), 914 properties, 360 typed mutants | cc-by-4.0 | **A**, **G**, B, C, H | Unlikely (synthetic) | **Use now**; split by `family_id` |
| 4 | [architect-ubc-capstone/rtl-augmented-v3](https://huggingface.co/datasets/architect-ubc-capstone/rtl-augmented-v3) | 948 injected bugs in 49 repos | mit (upstream varies) | **B, C, D** | Unlikely | **Use now**, after stripping comments (label leaks) |
| 5 | [HPAI-BSC/SystemVerilog-SFT-86k](https://huggingface.co/datasets/HPAI-BSC/SystemVerilog-SFT-86k) | 85,701 | none stated | H (test plans); compiled RTL | **Likely** (24/1000 OpenTitan) | **Use now, filtered** |
| 6 | [wilyub/VeriThoughtsTrainSet](https://huggingface.co/datasets/wilyub/VeriThoughtsTrainSet) | 20,173 (10,104 not equivalent) | mit | B (implementation vs reference), H | Low (1/500 OR1200) | Use later |
| 7 | [scale-lab/MetRex](https://huggingface.co/datasets/scale-lab/MetRex) | 25,868 | bsd-3-clause | Clean synthesizable RTL, H | Unlikely (0/1000) | **Use now** as a clean module pool |
| 8 | [theepicflyer/openrtlset-permissive-verified](https://huggingface.co/datasets/theepicflyer/openrtlset-permissive-verified) | 13,625 | cc-by-4.0 | Lint-clean RTL with repo field, H | Unlikely | **Use now** as a clean module pool |
| 9 | [ESCAD/OpenRTLSet](https://huggingface.co/datasets/ESCAD/OpenRTLSet) | 131,309 | apache-2.0 card; 52% GPL rows | Raw RTL, H, `Repo_url` | Unlikely (0/2000) | Use later |
| 10 | [bnadimi/PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) | 692,238 | cc-by-nc-sa-4.0 | H, raw RTL | Low (Ibex and PULP rows exist) | Use later, filtered |
| 11 | [GaTech-EIC/MG-Verilog](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog) | ~11K (paper; viewer broken) | mit | H; block summaries | Unknown | Use later |
| 12 | [SKLP-EDA-LAB/SemSpec-RTL-107](https://huggingface.co/datasets/SKLP-EDA-LAB/SemSpec-RTL-107) | 107 bug cases, 34 designs | none stated | B, D (eval-sized) | Unknown | Use later as a small extra D eval |
| 13 | [HPAI-BSC/SystemVerilog-RL-12k](https://huggingface.co/datasets/HPAI-BSC/SystemVerilog-RL-12k) | 11,902 (card) | none stated | Weak B/G: mutants are gate-level netlists | **Likely** | Probably skip |
| 14 | [zhuyaoyu/CodeV-R1-dataset](https://huggingface.co/datasets/zhuyaoyu/CodeV-R1-dataset) | 87,321 | cc-by-sa-4.0 | H, raw RTL | Unlikely | Use later |
| 15 | [yang-z/CodeV-All-dataset](https://huggingface.co/datasets/yang-z/CodeV-All-dataset) | 184,060 | apache-2.0 | H, raw RTL | **Likely** in the Chisel part | Use later, Verilog file only |
| 16 | [henryen/origen_dataset_instruction](https://huggingface.co/datasets/henryen/origen_dataset_instruction) | 222,075 | gpl-3.0 | H, raw RTL | Unlikely | Use later |
| 17 | [ahmedallam/RTL-Repo](https://huggingface.co/datasets/ahmedallam/RTL-Repo) | 4,098 | none stated | Multi-file repo context | Low | Use later |
| 18 | [shailja/Verilog_GitHub](https://huggingface.co/datasets/shailja/Verilog_GitHub) | 108,971 | mit tag / OpenRAIL-M in card | Raw RTL | **Likely** (OR1200 in 26/2000) | Skip, or dedup and filter first |
| 19 | [ytliu99/TroLL-Logic-Locking-based-Hardware-Trojans](https://huggingface.co/datasets/ytliu99/TroLL-Logic-Locking-based-Hardware-Trojans) | ~2,665 netlists, 3 base designs | cc-by-4.0 | T, gate-level only | Unlikely | Skip for now |
| 20 | [KSU-HW-SEC/Fixbench-RTL](https://huggingface.co/datasets/KSU-HW-SEC/Fixbench-RTL) | 100 pairs | cc-by-4.0 | B, C, D | **Certain** (24/100 test-family code) | Skip |

## Key findings

1. **Best A/G labels are small but formally checked.** HierSVA (342 BaseJump STL modules, SVAs proven
   non-vacuous with VC Formal per the card, 28 buggy variants with line, diff and violated property) and
   EquivSVA (120 families × 4 equivalent RTL variants, gold properties, 3 typed mutants each). Neither
   ships a property × mutant result matrix; we compute it with SymbiYosys.
2. **CodeV-SVA's unfiltered 324K file is the scalable A source.** Its SVAs were generated by DeepSeek-V3.1
   from the RTL and never checked against the RTL; some are plainly false (seen: `$stable(counter_up)` on a
   free-running counter). Running SymbiYosys gives natural violated/holds labels. Measure the open-source
   SVA parse rate on ~1K samples first (`|=>`, `##`, `disable iff`, `$past` are common).
3. **Label leaks to strip:** rtl-augmented-v3 injects comments such as `// Force unknown — makes missing reset
   observable`; dpo_verilog_buggy has `// Introduced bug:`; TroLL names nets `trigger`/`payload_net`.
   rtl-bug-fix-sft's "fix" diffs are reversed (golden→buggy) in 163/171 rows.
4. **Card claims contradicted by the data:** shailja/Verilog_GitHub is not deduplicated (18 unique of the
   first 100 rows) and has rows up to 452K chars; OpenRTLSet's rows are 52% GPL; CVDP v1.1.0 reference
   solutions are empty; SystemVerilog-RL-12k mutants are post-synthesis netlists without mutation types.
5. **No clean HF source for E (hardware CWE), F (security category), CDC/lint or Trojans in RTL.** Every
   hardware-security bug set found is built from the test families. We generate E/F/T data ourselves
   (Phase 5); HierSVA's `bsg_async_*` modules are a good multi-clock seed.

## KSU-HW-SEC (checked 2026-10-15)

All 12 sets on https://huggingface.co/KSU-HW-SEC. Useful later: SynthAgentic-Eval (MIT; 32,320 LLM-written
designs for 202 VerilogEval/RTLLM tasks with prompt, code and testbench pass/fail, about 24% fail: a "meets its
spec?" question). Maybe: 5W1HHardwareDebug (MIT, auto-gated; 810 semi-synthetic bugs from 8 small non-test repos).
No: LLM4SecHW-OSHD (auto-gated; bug-fix PRs from Pulp, OpenPiton, CVA6, Ibex, OpenTitan), issue_benchmark (a
400 KB sample is ~99% Ibex), Fixbench-RTL, hardware_code_and_sec_small/median/large (0.5-3.6M unlabeled
pretraining documents, 2.9-45 GB, auto-gated), Verilog_code, Structure_data, PPA_data (no bug or property labels),
food-science-llm-protocol.

## Do not train on (built from test families)

henryen/hwe-bench (the test set itself), rtl-debug-bench/rtl-debug-bench, Zeeh-Lin/RtlDebugBenchDataset,
KSU-HW-SEC/issue_benchmark, KSU-HW-SEC/Fixbench-RTL, vyomaalabs/verixa-dataset, hasankursun/soc-builder-rtl-v1,
billshockley/titanbench, tapeout-labs/hir-relationships, OvertureMMXXI/HDL-RepoBench-Sample,
KlaraHDL/hardware_code_and_sec_small, nvidia/cvdp-benchmark-dataset (agentic subset), rtl-llm/chisel-verilog-pairs.

## Skip (junk or off-target)

HadiJahanirad/HardwareTrojanVulnerabilityAnalysisDataset (a RAR of JPG images), verify-ppt/marin-starcoderdata_*
(pre-tokenized shards without repo metadata), TJ-24/SVADataset (tautological assertions), Jalik/* (no RTL;
`.pkl` files unsafe to load), matthewdelorenzo/dpo_verilog_buggy, hardware-fab/Chameleon (power traces).
Gated and not inspected: KSU-HW-SEC/LLM4SecHW-OSHD and the bigcode the-stack Verilog/SystemVerilog subsets.

## Practical notes

- **Decontaminate every GitHub-derived corpus** before training: drop rows matching `ibex_*`, `cva6`,
  `ariane_*`, `cv32e40p_*`, PULP `riscv_*`, `or1200_*`, `caliptra_*`, `prim_*`, `tlul_*`, "lowRISC contributors",
  "ETH Zurich and University of Bologna", `pulp-platform`, `openpiton`/`piton`. Prefer sources with repo
  metadata for exact filtering.
- **A and G:** start from HierSVA and EquivSVA, then scale with CodeV-SVA through SymbiYosys.
- **B, C and D:** rtl-augmented-v3 (comments stripped) plus our own mutants on MetRex /
  openrtlset-permissive-verified modules; SemSpec-RTL-107 as a sanity eval.
