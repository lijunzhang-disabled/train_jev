"""Render an example (spec/data_format.md) into token ids plus the positions the model reads answers from."""
import bisect

SYSTEM_PROMPT = ("You are an expert hardware design and verification engineer. "
                 "Read the RTL, then answer every question with one option.")
MARKER = "// [block {}]"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NONE = "none"


def code_with_markers(code, blocks):
    """The code with a marker comment after the last line of each block."""
    lines = code.split("\n")
    out = []
    for j, (start, end) in enumerate(blocks):
        out.extend(lines[start - 1:end])
        out.append(MARKER.format(j))
    return "\n".join(out)


def user_message(example, marked_code):
    parts = [f"File: {example['state']['path']}", "```systemverilog", marked_code, "```", "", "Questions:"]
    for i, q in enumerate(example["questions"], 1):
        parts.append(f"Q{i}. {q['text']}")
        if q["type"] == "choice":
            parts.extend(f"{LETTERS[k]}) {opt}" for k, opt in enumerate(q["options"]))
        else:
            parts.append("Answer with a block number, or none.")
    parts += ["", "Reply with one answer per question."]
    return "\n".join(parts)


def option_names(question, n_blocks):
    if question["type"] == "choice":
        return list(question["options"])
    return [str(j) for j in range(n_blocks)] + [NONE]


class Renderer:
    """Turns examples into model inputs. Needs a fast (offset-mapping) Hugging Face tokenizer."""

    def __init__(self, tokenizer, max_seq_tokens):
        self.tok = tokenizer
        self.max_seq_tokens = max_seq_tokens
        self.letter_ids = []
        for letter in LETTERS:
            ids = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"option letter {letter!r} is not a single token for this tokenizer")
            self.letter_ids.append(ids[0])

    def prompt(self, example):
        marked = code_with_markers(example["state"]["code"], example["state"]["blocks"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(example, marked)}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return text, marked

    def render(self, example):
        """Returns a dict with input_ids and per-question readout info, or None if it is too long."""
        text, marked = self.prompt(example)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        starts = [s for s, _ in enc["offset_mapping"]]

        blocks = example["state"]["blocks"]
        marker_pos, cursor = [], text.index(marked)
        for j in range(len(blocks)):
            m = MARKER.format(j)
            k = text.index(m, cursor)
            cursor = k + len(m)
            marker_pos.append(bisect.bisect_right(starts, cursor - 1) - 1)  # token holding the marker's last char

        labels = example.get("labels") or {}
        questions = []
        for i, q in enumerate(example["questions"]):
            ids += self.tok(("\n" if i else "") + f"Q{i + 1}:", add_special_tokens=False)["input_ids"]
            names = option_names(q, len(blocks))
            item = {"id": q["id"], "type": q["type"], "pos": len(ids) - 1, "options": names}
            if q["type"] == "choice":
                item["option_ids"] = self.letter_ids[:len(names)]
            else:
                item["marker_pos"] = marker_pos
            label = labels.get(q["id"])
            if label is not None:
                unknown = set(label["dist"]) - set(names)
                if unknown:
                    raise ValueError(f"{example['id']} {q['id']}: label keys {sorted(unknown)} not in options")
                total = sum(label["dist"].values())
                item["target"] = [label["dist"].get(n, 0.0) / total for n in names]
                item["weight"] = float(label.get("weight", 1.0))
            questions.append(item)

        if len(ids) > self.max_seq_tokens:
            return None
        return {"id": example["id"], "group": example.get("group", example["id"]), "input_ids": ids,
                "questions": questions}
