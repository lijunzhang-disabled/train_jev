# Data format v0

One JSON object per line. `tools/build_phase0_data.py` writes these from `data/benchmarks/benchmarks.jsonl`, and
`jev/format.py` turns each one into a single model input.

```json
{
  "id": "hwe/lowRISC__opentitan/pr-20304/0/buggy",
  "group": "hwe/lowRISC__opentitan/pr-20304",
  "source": "hwe-bench",
  "kind": "buggy",
  "state": {
    "path": "hw/ip/edn/rtl/edn_main_sm.sv",
    "code": "module edn_main_sm ...\n...",
    "blocks": [[1, 16], [17, 32], [33, 41]]
  },
  "questions": [
    {"id": "bug_present", "type": "choice", "text": "Does this code contain a functional bug?", "options": ["yes", "no"]},
    {"id": "bug_category", "type": "choice", "text": "What kind of bug does this code contain?",
     "options": ["no bug", "logic error", "..."]},
    {"id": "bug_location", "type": "pointer", "text": "Which block contains the bug?"}
  ],
  "labels": {
    "bug_present": {"dist": {"yes": 1.0}, "source": "construction", "weight": 1.0},
    "bug_category": {"dist": {"logic error": 1.0}, "source": "hwe-bench", "weight": 1.0},
    "bug_location": {"dist": {"2": 1.0}, "source": "construction", "weight": 1.0}
  }
}
```

| Field | Meaning |
| --- | --- |
| `id` | Unique example id |
| `group` | Examples from the same bug share a group; splits never separate a group |
| `kind` | `buggy`, `fixed` (same window after the fix), `elsewhere` (buggy file, away from the fix), `property` |
| `state.code` | The RTL window shown to the model (LF line endings) |
| `state.blocks` | 1-based, inclusive line ranges of `code`; the candidates for pointer questions |
| `questions` | Asked in order; `choice` questions list `options`, `pointer` questions point at a block or `none` |
| `labels.<qid>.dist` | Target distribution: option text for `choice`, block index (as a string) or `none` for `pointer` |
| `labels.<qid>.weight` | Loss weight (0.5 for `elsewhere` examples, whose "no bug" label is less certain) |

Every hwe-bench example asks the same three questions in the same order, so the question list itself
never hints at the answer. `bug_present` comes first because the model reads left to right: its answer
slot cannot see the later questions.

## How the model reads it

`jev/format.py` renders the example with the tokenizer's chat template (Qwen3, thinking disabled):

- The code goes into the user message, with a marker comment `// [block j]` after the last line of block `j`.
- The questions follow, each choice option labeled `A)`, `B)`, ...
- The assistant turn is `Q1:` `Q2:` ... with no answers written. The model reads the whole input once.
- **Choice questions:** the logits at the last token of `Qi:` restricted to the option-letter tokens
  (` A`, ` B`, ...). These come from the pretrained LM head, so they work zero-shot.
- **Pointer questions:** a small trained head scores each block marker's hidden state against the hidden
  state at `Qi:`, plus a learned `none` vector. The markers come after their block, so each marker has
  already read its block.

Training minimizes the cross-entropy between each question's softmax and its target distribution
(a proper scoring rule), weighted by `labels.<qid>.weight`.
