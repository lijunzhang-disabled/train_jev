#!/usr/bin/env python3
"""Inject one-edit bugs into clean Verilog designs, label each mutant with an equivalence check, and write
module-level examples (spec/data_format.md).

    python tools/build_mutants.py --pool data/raw/metrex/metrex.json --limit 1000 --jobs 8 --out data/mutants_pilot
    python tools/build_mutants.py --pool data/raw/metrex/metrex.json data/raw/openrtlset/data.jsonl \
        --limit 100000 --jobs 16 --out data/mutants_v1          # every eligible design of both pools

Needs Yosys on PATH (OSS CAD Suite, guide step 5; --yosys picks another binary) and `transformers` for the
tokenizer. Designs are tried in a fixed random order; --limit counts designs sent to Yosys.

Per design, in a worker process:
  1. Yosys parses and elaborates it and picks the top module. Designs that fail, that instantiate a module
     the file doesn't define, or whose top has no inputs or no outputs are skipped.
  2. Up to --mutants-per-design operators that have a site in the top's hierarchy are drawn, one site each
     (OPERATORS below). A mutant changes one spot, usually inside one line; comments are never touched.
  3. A miter of the original and the mutant goes to `sat -tempinduct` with all registers starting at zero:
     a counterexample means the mutant behaves differently (a bug); an induction proof means it can't
     (kind "equivalent", labeled "no bug"); a step limit, solver timeout or error drops the mutant, and so
     does a mutant that adds multiple drivers or a logic loop (Yosys `check`), where SAT isn't sound.
Examples per design: the original ("clean"), its behavior-changing mutants ("buggy"; bug location is the
block holding the edit) and its proven-equivalent mutants ("equivalent"). A design and its mutants share
a group, so they train in the same step and land in the same split. Mutants get a uniform bug_category
target over the five bug kinds at weight 0.5: hwe-bench's categories don't map onto mutation operators,
but "no bug" is wrong for them.

Writes train.jsonl, val.jsonl, checks.jsonl (every Yosys run, for measuring) and stats.json.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_phase0_data import CATEGORIES, HWE_QUESTIONS, NO_BUG, make_blocks  # noqa: E402
from jev.format import NONE  # noqa: E402

OPERATORS = {
    "relop_flip": "comparison changed: == and !=, < and <=, > and >=",
    "logic_swap": "&& and ||, or binary & and |, swapped",
    "arith_swap": "binary + and - swapped",
    "constant": "a constant in an expression or condition off by one (a 1-bit constant flips)",
    "reset_value": "a constant assigned under a reset condition changed (zero to one, otherwise to zero)",
    "cond_negate": "an if condition on a single signal inverted (! added or removed)",
    "drop_term": "one operand of an if condition's && or || chain removed",
    "drop_not": "a unary ! or ~ removed",
    "state_swap": "a named constant (state, opcode) replaced by another from the same parameter group",
    "signal_swap": "a signal read replaced by another signal of the same declared width",
    "index_shift": "a constant bit select a[N] changed to a[N-1]",
    "drop_assignment": "an assignment statement on its own line deleted",
}
TEST_FAMILIES = re.compile(r"ibex|cva6|ariane|pulpissimo|openpiton|or1200|caliptra|opentitan|lowrisc|\bprim_|"
                           r"hack_?@?dac", re.I)
TESTBENCH = re.compile(r"\$(finish|stop|monitor|dumpfile|dumpvars)\b")
CLOCK = re.compile(r"clk|clock", re.I)
RESET = re.compile(r"rst|reset", re.I)

TOKEN_RE = re.compile("|".join([
    r"(?P<comment>//[^\n]*|/\*.*?\*/)",
    r'(?P<string>"(?:\\.|[^"\\\n])*")',
    r"(?P<directive>`[A-Za-z_]\w*)",
    r"(?P<number>(?:\d[\d_]*[ \t]*)?'[sS]?[bBoOdDhH][ \t]*[0-9a-fA-F_xXzZ?]+|'[01xXzZ](?![\w'])|\d[\d_]*(?:\.\d+)?)",
    r"(?P<ident>[A-Za-z_][\w$]*|\\\S+)",
    r"(?P<system>\$[A-Za-z_][\w$]*)",
    r"(?P<op><<<|>>>|===|!==|==|!=|<=|>=|&&|\|\||<<|>>|~&|~\||~\^|\^~|->|\+:|-:|::|\*\*|[-+*/%<>=!~&|^?:;,.#@(){}\[\]'])",
    r"(?P<space>\s+)",
    r"(?P<other>.)",
]), re.S)
KEYWORDS = set("""always always_comb always_ff always_latch and assign automatic begin buf bufif0 bufif1 case casex
casez default defparam disable else end endcase endfunction endgenerate endmodule endtask enum final for forever fork
function generate genvar if initial inout input int integer join localparam logic module nand negedge nor not notif0
notif1 or output packed parameter posedge priority real reg repeat signed struct supply0 supply1 task time tri
typedef unique unsigned var wand while wire wor xnor xor bit byte shortint longint""".split())
DIRECTION_KW = {"input", "output", "inout", "parameter", "localparam", "genvar", "typedef"}
DECL_KW = DIRECTION_KW | {"wire", "reg", "logic", "integer", "real", "time", "bit", "byte", "int", "shortint",
                          "longint", "tri", "supply0", "supply1", "wand", "wor", "var"}
BEHAV_KW = {"always", "always_ff", "always_comb", "always_latch"}
RELOP = {"==": "!=", "!=": "==", "<": "<=", "<=": "<", ">": ">=", ">=": ">"}


class Tok:
    __slots__ = ("kind", "text", "start", "end", "module", "region", "role", "bracket", "stmt", "reset", "span")

    def __init__(self, kind, text, start, end):
        self.kind, self.text, self.start, self.end = kind, text, start, end
        self.module, self.region, self.role, self.bracket, self.stmt, self.reset, self.span = (
            None, "other", "other", 0, 0, False, None)


def lex(text):
    """The tokens of the source, without whitespace and comments."""
    return [Tok(m.lastgroup, m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)
            if m.lastgroup not in ("space", "comment")]


def operand_end(t):
    return (t.kind in ("ident", "number", "string", "system") and t.text not in KEYWORDS) or t.text in (")", "]", "}")


def annotate(toks):
    """Gives each token its module, module-item region (always, assign, decl, header, ...) and role: "lhs" and
    "rhs" of a behavioral assignment, "assign" for the operator itself, "cond" inside an if/case/while
    condition, "label" for case item labels, "decl", "skip" (event controls, for headers, delays), "other".
    Returns the condition spans: (kind, index of "(", index of ")"). A heuristic: Yosys checks every mutant."""
    module, region = None, "other"
    be = paren = bracket = 0
    spans, cond_spans = [], []        # open: [kind, paren level, open index, has a reset signal]
    pending = None
    stmt, stmt_decl, assigned, at_start = 0, False, False, True
    case_levels, label = [], False
    reset_next, reset_until = False, None
    star_skip = False
    for i, t in enumerate(toks):
        x = t.text
        nxt = toks[i + 1].text if i + 1 < len(toks) else ""
        if x == "module" and t.kind == "ident":
            module = nxt
            region, be, paren, bracket, spans, case_levels, label = "header", 0, 0, 0, [], [], False
            stmt += 1
            stmt_decl, assigned, at_start, reset_until, reset_next = True, False, False, None, False
        if x == "endmodule":
            t.module, t.role = module, "other"
            module, region, at_start = None, "other", True
            continue

        if reset_next:
            reset_until = ("end", be) if x == "begin" else ("semi", be)
            reset_next = False
        if at_start and x not in ("begin", "end", "else", "endcase", ";") and not (t.kind == "op" and x != "#"):
            stmt += 1
            stmt_decl, assigned, at_start = x in DECL_KW, False, False
            if be == 0 and paren == 0:
                if x in BEHAV_KW:
                    region = "always"
                elif x in ("initial", "final"):
                    region = "initial"
                elif x == "assign":
                    region = "assign"
                elif x in ("function", "task"):
                    region = "function"
                elif x in DECL_KW:
                    region = "decl"
                elif t.kind == "ident" and x not in KEYWORDS:
                    region = "instance"
        t.module, t.region, t.bracket, t.stmt = module, region, bracket, stmt
        t.reset = reset_until is not None

        if spans:
            t.role = "cond" if spans[-1][0] in ("if", "while", "case") else "skip"
            t.span = spans[-1][2] if t.role == "cond" else None
            if t.role == "cond" and t.kind == "ident" and RESET.search(x):
                spans[-1][3] = True
        elif label:
            t.role = "label"
        elif region in ("always", "assign", "function") and not stmt_decl:
            t.role = "rhs" if assigned else "lhs"
        elif region in ("header", "decl") or stmt_decl:
            t.role = "decl"
        else:
            t.role = "other"

        if star_skip:
            star_skip, t.role, at_start = False, "skip", True
            continue
        if x == "(":
            paren += 1
            if pending:
                spans.append([pending, paren, i, False])
                pending = None
        elif x == ")":
            if spans and spans[-1][1] == paren:
                kind, _, opened, has_reset = spans.pop()
                if kind in ("if", "while", "case"):
                    cond_spans.append((kind, opened, i))
                if kind == "case":
                    be += 1
                    case_levels.append(be)
                    label = True
                if kind in ("if", "while", "for", "event", "case"):
                    at_start = True
                if kind == "if" and has_reset and region == "always":
                    reset_next = True
            paren -= 1
        elif x == "[":
            bracket += 1
        elif x == "]":
            bracket -= 1
        elif x == "if" or x == "while":
            pending = x
        elif x in ("case", "casez", "casex"):
            pending = "case"
        elif x == "for":
            pending = "for"
        elif x == "@":
            pending = "event"
            t.role = "skip"
            if nxt == "*":
                star_skip, pending = True, None
        elif x == "#":
            t.role = "skip"
            if nxt == "(":
                pending = "delay"
            elif i + 1 < len(toks):
                toks[i + 1].role = "skip"
        elif x in ("begin", "fork"):
            be += 1
            at_start, label = True, False
        elif x in ("end", "join"):
            be -= 1
            at_start = True
            if case_levels and be == case_levels[-1]:
                label = True
            if reset_until and reset_until[0] == "end" and be == reset_until[1]:
                reset_until = None
        elif x == "endcase":
            be -= 1
            if case_levels:
                case_levels.pop()
            at_start, label = True, bool(case_levels and be == case_levels[-1])
        elif x in ("function", "task") and be == 0:
            be += 1
        elif x in ("endfunction", "endtask"):
            be = max(0, be - 1)
            at_start, region = True, "other"
        elif x == "else":
            at_start, label = True, False
        elif x in BEHAV_KW or x in ("initial", "final"):
            at_start = True
        elif x == ";" and not spans:
            at_start, assigned, stmt_decl = True, False, False
            if case_levels and be == case_levels[-1]:
                label = True
            if be == 0 and region in ("assign", "decl", "instance", "header"):
                region = "other"
            if reset_until and reset_until[0] == "semi" and be == reset_until[1]:
                reset_until = None
        elif x == ":" and label and paren == 0 and bracket == 0:
            label, at_start = False, True
        elif x in ("=", "<=") and not spans and paren == 0 and bracket == 0 and not assigned \
                and t.role == "lhs":
            assigned, t.role = True, "assign"
    return cond_spans


def collect_decls(toks):
    """Per module: declared signals with their packed range text ("" for one bit), arrays, and groups of
    named constants (consecutive parameter/localparam statements, or one enum)."""
    sigs = collections.defaultdict(dict)
    arrays = collections.defaultdict(set)
    groups = collections.defaultdict(list)
    by_stmt = collections.defaultdict(list)
    for t in toks:
        if t.role == "decl" and t.module:
            by_stmt[(t.module, t.stmt)].append(t)
    last_param = {}
    for (module, stmt), ts in sorted(by_stmt.items(), key=lambda kv: kv[1][0].start):
        kind, rng, after_name, depth, init_depth, in_enum = None, "", False, 0, None, False
        enum_names, i = [], 0
        is_param = any(t.text in ("parameter", "localparam") for t in ts)
        names = []
        while i < len(ts):
            x = ts[i].text
            if init_depth is not None:
                if x in "([{":
                    depth += 1
                elif x in ")]}":
                    depth -= 1
                if (x in (",", ";") and depth == init_depth) or depth < init_depth:
                    init_depth = None
                    after_name = False
                i += 1
                continue
            if x == "enum":
                in_enum = True
            elif x == "{" and in_enum:
                depth += 1
                j = i + 1
                while j < len(ts) and ts[j].text != "}":
                    if ts[j].kind == "ident" and ts[j].text not in KEYWORDS and ts[j - 1].text in ("{", ","):
                        enum_names.append(ts[j].text)
                    j += 1
                i = j + 1
                depth -= 1
                in_enum = False
                continue
            elif x in DECL_KW:
                if x in DIRECTION_KW or kind is None:
                    kind = x
                rng, after_name = "", False
            elif x == "[":
                j, d = i, 0
                while j < len(ts):
                    d += ts[j].text == "["
                    d -= ts[j].text == "]"
                    if d == 0:
                        break
                    j += 1
                if after_name and names:
                    arrays[module].add(names[-1])
                else:
                    rng = " ".join(t.text for t in ts[i:j + 1])
                i = j + 1
                continue
            elif x in "({":
                depth += 1
            elif x in ")}":
                depth -= 1
            elif x == "=":
                init_depth = depth
            elif x == ",":
                after_name = False
            elif ts[i].kind == "ident" and x not in KEYWORDS and kind is not None and not after_name:
                nxt = ts[i + 1].text if i + 1 < len(ts) else ";"
                if nxt in (",", ";", ")", "=", "["):
                    names.append(x)
                    if kind not in ("parameter", "localparam", "typedef", "genvar"):
                        sigs[module][x] = rng
                    after_name = True
            i += 1
        if enum_names:
            groups[module].append(enum_names)
        if is_param and names:
            if last_param.get(module) == stmt - 1 and groups[module]:
                groups[module][-1].extend(names)
            else:
                groups[module].append(list(names))
            last_param[module] = stmt
    return sigs, arrays, groups


def parse_number(s):
    """(size or None, base letter or None, digits, value) for a plain sized or decimal literal, else None."""
    m = re.fullmatch(r"(?:(\d+)[ \t]*)?'([sS]?)([bBoOdDhH])[ \t]*([0-9a-fA-F]+)", s)
    if m:
        size, signed, base, digits = m.groups()
        radix = {"b": 2, "o": 8, "d": 10, "h": 16}[base.lower()]
        try:
            return (int(size) if size else None, signed + base, digits, int(digits, radix))
        except ValueError:
            return None
    if re.fullmatch(r"\d+", s):
        return None, None, s, int(s)
    return None


def format_number(s, value):
    """The literal s rewritten to hold value, in the same style, or None if it doesn't fit its size."""
    p = parse_number(s)
    if p is None or value < 0:
        return None
    size, base, digits, _ = p
    if size is not None and value >= 2 ** size:
        return None
    if base is None:
        return str(value)
    radix = {"b": 2, "o": 8, "d": 10, "h": 16}[base[-1].lower()]
    out = ""
    v = value
    while v:
        out = "0123456789abcdef"[v % radix] + out
        v //= radix
    out = out or "0"
    if digits != digits.lower():
        out = out.upper()
    if len(digits) > 1 and digits[0] == "0" or radix in (2, 16):
        out = out.rjust(len(digits), "0")
    head = s[:s.index("'")]
    return f"{head}'{base}{out}"


class Edit:
    __slots__ = ("op", "start", "end", "repl")

    def __init__(self, op, start, end, repl):
        self.op, self.start, self.end, self.repl = op, start, end, repl


def find_sites(text, toks, cond_spans, modules, rng):
    """Every candidate edit, by operator, inside the given modules."""
    sigs, arrays, groups = collect_decls(toks)
    group_of = {m: {n: g for g in gs for n in g} for m, gs in groups.items()}
    stmt_names = collections.defaultdict(set)
    for t in toks:
        if t.kind == "ident":
            stmt_names[(t.module, t.stmt)].add(t.text)
    by_width = collections.defaultdict(list)   # (module, range text) -> signals a read may be swapped to
    for m, names in sigs.items():
        for n, w in sorted(names.items()):
            if n not in arrays[m] and not CLOCK.search(n) and not RESET.search(n):
                by_width[(m, w)].append(n)
    sites = collections.defaultdict(list)

    def add(op, t, repl):
        sites[op].append(Edit(op, t.start, t.end, repl))

    cond_first = {}   # index of the first token of a simple if condition -> span
    for kind, o, c in cond_spans:
        if kind != "if" or toks[o].module not in modules:
            continue
        inner = toks[o + 1:c]
        texts = [t.text for t in inner]
        simple = len(inner) == 1 or (len(inner) == 4 and texts[1] == "[" and texts[3] == "]")
        if inner and inner[0].kind == "ident" and simple and inner[0].text not in KEYWORDS:
            sites["cond_negate"].append(Edit("cond_negate", inner[0].start, inner[0].start, "!"))
        elif texts and texts[0] in ("!", "~") and len(inner) >= 2 and inner[1].kind == "ident" and \
                (len(inner) == 2 or (len(inner) == 5 and texts[2] == "[" and texts[4] == "]")):
            sites["cond_negate"].append(Edit("cond_negate", inner[0].start, inner[0].end, ""))
            cond_first[o + 1] = True
        # top-level && / || chains
        depth, ops_at, ops = 0, [], set()
        for k in range(o + 1, c):
            x = toks[k].text
            if x in "([{":
                depth += 1
            elif x in ")]}":
                depth -= 1
            elif depth == 0 and x in ("&&", "||"):
                ops_at.append(k)
                ops.add(x)
            elif depth == 0 and x in ("?", ":"):
                ops.add("?")
        if ops_at and len(ops) == 1:
            bounds = [o + 1] + [k + 1 for k in ops_at]
            ends = [k - 1 for k in ops_at] + [c - 1]
            j = rng.randrange(len(bounds))
            if j == 0:
                sites["drop_term"].append(Edit("drop_term", toks[bounds[0]].start, toks[bounds[1]].start, ""))
            else:
                sites["drop_term"].append(Edit("drop_term", toks[ends[j - 1]].end, toks[ends[j]].end, ""))

    for k, t in enumerate(toks):
        if t.module not in modules:
            continue
        prev = toks[k - 1] if k else None
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if t.role in ("rhs", "cond") and t.bracket == 0:
            x = t.text
            if x in RELOP and t.kind == "op":
                add("relop_flip", t, RELOP[x])
            if x in ("&&", "||"):
                add("logic_swap", t, "||" if x == "&&" else "&&")
            if x in ("&", "|") and prev and operand_end(prev):
                add("logic_swap", t, "|" if x == "&" else "&")
            if x in ("+", "-") and prev and operand_end(prev):
                add("arith_swap", t, "-" if x == "+" else "+")
            if x in ("!", "~") and not (prev and operand_end(prev)) and nxt and \
                    (nxt.kind == "ident" or nxt.text == "(") and k not in cond_first:
                add("drop_not", t, "")
            if t.kind == "number" and not (nxt and nxt.text == "{"):
                p = parse_number(x)
                fill = re.fullmatch(r"'([01])", x)
                if t.reset and t.role == "rhs":
                    if fill:
                        add("reset_value", t, "'1" if x == "'0" else "'0")
                    elif p:
                        new = format_number(x, 1 if p[3] == 0 else 0)
                        if new:
                            add("reset_value", t, new)
                elif p:
                    size, _, _, v = p
                    if size == 1:
                        new = format_number(x, 1 - v)
                    else:  # never 1 -> 0: "+ 0" or ">> 0" would give the edit away
                        new = format_number(x, v + 1 if v <= 1 or rng.random() < 0.5 else v - 1)
                        new = new or (format_number(x, v - 1) if v > 1 else None)
                    if new and new != x:
                        add("constant", t, new)
            if t.kind == "ident" and x not in KEYWORDS and nxt and nxt.text != "(":
                grp = group_of.get(t.module, {}).get(x)
                if grp and len(grp) >= 3 and prev and (
                        (prev.role == "assign" and nxt.text == ";") or prev.text in ("==", "!=")):
                    add("state_swap", t, rng.choice([n for n in grp if n != x]))
                width = sigs.get(t.module, {}).get(x)
                if width is not None and x not in arrays[t.module] and not CLOCK.search(x) and not RESET.search(x):
                    taken = stmt_names[(t.module, t.stmt)]  # no "b - b" or "x <= x"-style self-references
                    pool = by_width[(t.module, width)]
                    if len(pool) > len(taken) or any(n not in taken for n in pool):
                        pool = [n for n in pool if n not in taken]
                        if pool:
                            add("signal_swap", t, rng.choice(pool))
        if t.role in ("lhs", "rhs", "cond") and t.bracket == 1 and t.kind == "number" and prev and \
                prev.text == "[" and nxt and nxt.text == "]":
            p = parse_number(t.text)
            if p and p[3] > 0:
                new = format_number(t.text, p[3] - 1)
                if new:
                    add("index_shift", t, new)

    # whole assignment statements on their own line, inside a sequence of statements
    by_stmt = collections.defaultdict(list)
    for k, t in enumerate(toks):
        if t.region == "always" and t.module in modules and t.role in ("lhs", "assign", "rhs"):
            by_stmt[t.stmt].append(k)
    for ks in by_stmt.values():
        first, semi = ks[0], ks[-1]
        if toks[semi].text != ";" or not any(toks[k].role == "assign" for k in ks):
            continue
        if first == 0 or toks[first - 1].text not in (";", "begin", "end"):
            continue
        if semi + 1 < len(toks) and toks[semi + 1].text == "else":
            continue
        ls = text.rfind("\n", 0, toks[first].start) + 1
        le = text.find("\n", toks[semi].end)
        le = len(text) if le < 0 else le
        rest = text[toks[semi].end:le].strip()
        if text[ls:toks[first].start].strip() or (rest and not rest.startswith("//")):
            continue
        if le < len(text):
            sites["drop_assignment"].append(Edit("drop_assignment", ls, le + 1, ""))
    return sites


def apply(text, e):
    return text[:e.start] + e.repl + text[e.end:]


def edit_lines(text, e):
    """1-based lines of the mutant that hold the edit (for a deleted line: the lines on either side)."""
    first = text.count("\n", 0, e.start) + 1
    if e.op == "drop_assignment":
        n = apply(text, e).count("\n") + 1
        return sorted({max(1, first - 1), min(first, n)})
    return list(range(first, first + max(1, e.repl.count("\n") + 1)))


# ---------------------------------------------------------------- Yosys

# Undriven nets and x constants become 0 in both designs: left free, the solver would pick them separately for
# the original and the mutant and report differences that aren't there.
ELAB = "proc; flatten; memory; async2sync; setundef -undriven -zero; opt_clean"
# `check` problems that make the SAT model unsound: multiple drivers give contradictory constraints that hide
# differences (a moved `assign out[13]` came back "equivalent"), and logic loops have no defined value.
UNSOUND = re.compile(r"multiple conflicting drivers|found logic loop", re.I)


def read_and_remove(workdir, name):
    path = os.path.join(workdir, name)
    if not os.path.exists(path):
        return ""
    with open(path) as fh:
        text = fh.read()
    os.remove(path)
    return text


def run_yosys(yosys, workdir, script, timeout):
    with open(os.path.join(workdir, "run.ys"), "w") as fh:
        fh.write(script)
    t0 = time.time()
    try:
        p = subprocess.run(yosys + ["-q", "-s", "run.ys"], cwd=workdir, capture_output=True, text=True,
                           timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "", None
    return out, rc, time.time() - t0


def first_error(log):
    for line in log.split("\n"):
        if "ERROR" in line:
            return re.sub(r"\s+", " ", line.strip())[:300]
    return log.strip().split("\n")[-1][:300] if log.strip() else "no output"


def parse_design(cfg, workdir):
    script = (f"read_verilog -sv design.v\nhierarchy -check -auto-top\nproc\nwrite_json hier.json\n"
              f"{ELAB.replace('proc; ', '')}\ntee -q -o check.log check\n")
    log, rc, secs = run_yosys(cfg["yosys"], workdir, script, cfg["timeout"])
    if rc is None:
        return {"status": "parse timeout"}, secs
    if rc != 0:
        return {"status": "parse error", "error": first_error(log)}, secs
    if UNSOUND.search(read_and_remove(workdir, "check.log")):
        return {"status": "multiple drivers or logic loop"}, secs
    with open(os.path.join(workdir, "hier.json")) as fh:
        mods = json.load(fh)["modules"]
    top = next((n for n, m in mods.items() if int(m.get("attributes", {}).get("top", "0"), 2)), None)
    if top is None:
        return {"status": "no top"}, secs
    dirs = collections.Counter(p["direction"] for p in mods[top]["ports"].values())
    if not dirs["input"] or not dirs["output"]:
        return {"status": "top has no inputs or no outputs"}, secs
    base = {(n.split("\\")[1] if n.startswith("$paramod") else n) for n in mods}
    return {"status": "ok", "top": top, "modules": sorted(base)}, secs


def check_mutant(cfg, workdir, top):
    script = "\n".join([
        "read_verilog -sv gold.v", f"hierarchy -check -top {top}", ELAB, f"rename {top} jev_gold", "design -stash gold",
        "read_verilog -sv gate.v", f"hierarchy -check -top {top}", ELAB, "tee -q -o check.log check",
        f"rename {top} jev_gate", "design -stash gate",
        "design -copy-from gold -as jev_gold jev_gold", "design -copy-from gate -as jev_gate jev_gate",
        # no -ignore_gold_x: without `sat -enable_undef` it makes every comparison vacuous
        "miter -equiv -flatten -make_assert jev_gold jev_gate jev_miter",
        "hierarchy -top jev_miter",
        f"tee -o sat.log sat -tempinduct -prove-asserts -set-init-zero -maxsteps {cfg['max_steps']} "
        f"-timeout {cfg['sat_timeout']} jev_miter", ""])
    log, rc, secs = run_yosys(cfg["yosys"], workdir, script, cfg["timeout"])
    sat = read_and_remove(workdir, "sat.log")
    if UNSOUND.search(read_and_remove(workdir, "check.log")):
        return "multiple drivers or logic loop", None, secs
    if rc is None:
        return "timeout", None, secs
    if "model found for base case: FAIL" in sat:
        return "differs", None, secs
    if "Induction step proven: SUCCESS" in sat:
        return "equivalent", None, secs
    if "TIMEOUT" in sat:
        return "sat timeout", None, secs
    if "Reached maximum number of time steps" in sat:
        return "step limit", None, secs
    if rc != 0:
        err = first_error(log)
        return ("mutant error" if "gate.v" in err else "error"), err, secs
    return "unknown", sat.strip().split("\n")[-1][:200] if sat.strip() else None, secs


def work(task):
    gid, text, cfg = task
    rng = random.Random(f"{cfg['seed']}/{gid}")
    out = {"group": gid, "lines": text.count("\n") + 1, "mutants": []}
    with tempfile.TemporaryDirectory(prefix="jevmut") as d:
        for name in ("design.v", "gold.v"):
            with open(os.path.join(d, name), "w") as fh:
                fh.write(text)
        info, secs = parse_design(cfg, d)
        out.update(parse=info["status"], parse_s=round(secs, 3))
        if info["status"] != "ok":
            out["error"] = info.get("error")
            return out
        out["top"] = top = info["top"]
        toks = lex(text)
        cond_spans = annotate(toks)
        sites = find_sites(text, toks, cond_spans, set(info["modules"]), rng)
        out["ops_available"] = sorted(sites)
        chosen = rng.sample(sorted(sites), min(cfg["per_design"], len(sites)))
        lines = text.split("\n")
        for op in chosen:
            e = rng.choice(sites[op])
            mutant = apply(text, e)
            if mutant == text:
                continue
            with open(os.path.join(d, "gate.v"), "w") as fh:
                fh.write(mutant)
            verdict, err, secs = check_mutant(cfg, d, top)
            ln = text.count("\n", 0, e.start)
            mlines = mutant.split("\n")
            out["mutants"].append({
                "op": op, "verdict": verdict, "seconds": round(secs, 3), "error": err,
                "edit": [e.start, e.end, e.repl], "lines": edit_lines(text, e),
                "before": lines[ln], "after": mlines[ln] if e.op != "drop_assignment" else ""})
    return out


# ---------------------------------------------------------------- examples

def normalize(src):
    lines = [line.rstrip() for line in src.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip("\n")


def code_key(text):
    return hashlib.sha256(" ".join(re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S).split()).encode()
                          ).hexdigest()


def pool_rows(path):
    """(group id, source text, split key, source name) per design. MetRex rows hold the source in RTL; openrtlset
    rows hold it in completion, inside a ```verilog fence, with the GitHub repo it came from."""
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            if "RTL" in row:
                yield f"{name}/{i}", row["RTL"], f"{name}/{i}", name
            else:
                code = re.sub(r"^\s*```[\w-]*\n|\n```\s*$", "", row["completion"])
                yield f"{name}/{row['id']}", code, f"{name}/{row['repo']}", name


def example(eid, group, kind, path, code, block_lines, present, location, meta=None, source="metrex-mutants"):
    lines = code.split("\n")
    blocks = make_blocks(len(lines), block_lines)
    if present == "yes":
        hit = sorted({k for ln in location for k, (s, e) in enumerate(blocks) if s <= ln <= e})
        loc = {str(k): 1.0 / len(hit) for k in hit}
        category = {"dist": {c: 1.0 for c in CATEGORIES.values()}, "source": "construction", "weight": 0.5}
    else:
        loc = {NONE: 1.0}
        category = {"dist": {NO_BUG: 1.0}, "source": "construction", "weight": 1.0}
    ex = {
        "id": eid, "group": group, "source": source, "kind": kind,
        "state": {"path": path, "code": code, "blocks": blocks},
        "questions": HWE_QUESTIONS,
        "labels": {
            "bug_present": {"dist": {present: 1.0}, "source": "equivalence check", "weight": 1.0},
            "bug_category": category,
            "bug_location": {"dist": loc, "source": "construction", "weight": 1.0},
        },
    }
    if meta:
        ex["mutation"] = meta
    return ex


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", nargs="+", default=[os.path.join(ROOT, "data", "raw", "metrex", "metrex.json")],
                    help="JSON-lines pools: MetRex (metrex.json) and/or openrtlset (openrtlset/data.jsonl)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mutants_pilot"))
    ap.add_argument("--limit", type=int, default=1000, help="designs to send to Yosys")
    ap.add_argument("--jobs", type=int, default=8, help="worker processes, one Yosys run each")
    ap.add_argument("--yosys", default="yosys", help="Yosys command, e.g. 'yosys' or 'yowasp-yosys'")
    ap.add_argument("--mutants-per-design", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20, help="sat -tempinduct step limit")
    ap.add_argument("--sat-timeout", type=int, default=30, help="seconds per SAT solver call")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per Yosys run")
    ap.add_argument("--min-lines", type=int, default=20)
    ap.add_argument("--max-code-tokens", type=int, default=3000)
    ap.add_argument("--max-seq-tokens", type=int, default=4096)
    ap.add_argument("--block-lines", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-8B"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from jev.format import Renderer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Renderer(tok, args.max_seq_tokens)
    yosys = shlex.split(args.yosys)
    version = subprocess.run(yosys + ["-V"], capture_output=True, text=True).stdout.strip()
    print(f"yosys: {version}", flush=True)

    # static filters, then a fixed random order
    pool = collections.Counter()
    seen, candidates, split_keys, sources = set(), [], {}, {}
    for path in args.pool:
        for gid, src, split_key, name in pool_rows(path):
            pool["designs"] += 1
            text = normalize(src)
            key = code_key(text)
            if key in seen:
                pool["skipped: duplicate"] += 1
                continue
            seen.add(key)
            if TEST_FAMILIES.search(text):
                pool["skipped: test-family name"] += 1
            elif TESTBENCH.search(text):
                pool["skipped: testbench"] += 1
            elif "`include" in text:
                pool["skipped: `include"] += 1
            elif not re.search(r"\b(always|always_ff|always_comb|assign)\b", text):
                pool["skipped: no always or assign"] += 1
            elif text.count("\n") + 1 < args.min_lines:
                pool["skipped: fewer than --min-lines"] += 1
            elif len(text) > 12 * args.max_code_tokens:  # sure to exceed it; spares the tokenizer the giants
                pool["skipped: more than --max-code-tokens"] += 1
            else:
                candidates.append((gid, text))
                split_keys[gid], sources[gid] = split_key, f"{name}-mutants"
    lengths = tok([t for _, t in candidates], add_special_tokens=False)["input_ids"]
    fits = [c for c, ids in zip(candidates, lengths) if len(ids) <= args.max_code_tokens]
    pool["skipped: more than --max-code-tokens"] += len(candidates) - len(fits)
    pool["eligible"] = len(fits)
    random.Random(args.seed).shuffle(fits)
    tasks = fits[:args.limit]
    print(f"pool: {dict(pool)}; sending {len(tasks)} designs to Yosys with {args.jobs} jobs", flush=True)

    cfg = {"yosys": yosys, "timeout": args.timeout, "sat_timeout": args.sat_timeout, "max_steps": args.max_steps,
           "per_design": args.mutants_per_design, "seed": args.seed}
    texts = dict(tasks)
    results, t0 = [], time.time()
    os.makedirs(args.out, exist_ok=True)
    with Pool(args.jobs) as workers, open(os.path.join(args.out, "checks.jsonl"), "w") as log:
        for r in workers.imap_unordered(work, [(g, t, cfg) for g, t in tasks], chunksize=1):
            results.append(r)
            log.write(json.dumps(r) + "\n")
            if len(results) % 50 == 0 or len(results) == len(tasks):
                n_mut = sum(len(x["mutants"]) for x in results)
                print(f"   {len(results)}/{len(tasks)} designs, {n_mut} mutants checked, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    wall = time.time() - t0

    splits = collections.defaultdict(list)
    lengths = collections.defaultdict(list)
    dropped = collections.Counter()
    for r in sorted(results, key=lambda x: x["group"]):
        if r["parse"] != "ok":
            continue
        g, text = r["group"], texts[r["group"]]
        path = f"{r['top']}.v"
        src = sources[g]
        exs = [example(f"{g}/clean", g, "clean", path, text, args.block_lines, "no", [], source=src)]
        for j, m in enumerate(r["mutants"]):
            meta = {k: m[k] for k in ("op", "before", "after")}
            mutant = text[:m["edit"][0]] + m["edit"][2] + text[m["edit"][1]:]
            if m["verdict"] == "differs":  # ids pair each mutant with /clean in jev.metrics.pair_accuracy
                exs.append(example(f"{g}/buggy-{j}", g, "buggy", path, mutant, args.block_lines, "yes", m["lines"],
                                   meta, source=src))
            elif m["verdict"] == "equivalent":
                exs.append(example(f"{g}/equivalent-{j}", g, "equivalent", path, mutant, args.block_lines, "no", [],
                                   meta, source=src))
        # split by repo where the pool records it, so one project never lands in both splits
        h = int(hashlib.sha256(split_keys[g].encode()).hexdigest(), 16) % 1000
        split = "val" if h < args.val_fraction * 1000 else "train"
        for ex in exs:
            rendered = renderer.render(ex)
            if rendered is None:
                dropped["rendered example too long"] += 1
                continue
            splits[split].append(ex)
            lengths[split].append(len(rendered["input_ids"]))

    for name in ("train", "val"):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for ex in splits[name]:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    parse = collections.Counter(r["parse"] for r in results)
    parse_errors = collections.Counter(re.sub(r"design\.v:\d+", "design.v:N", r["error"] or "")[:120]
                                       for r in results if r["parse"] == "parse error")
    by_op = collections.defaultdict(collections.Counter)
    available = collections.Counter(op for r in results for op in r.get("ops_available", []))
    secs = [m["seconds"] for r in results for m in r["mutants"]]
    for r in results:
        for m in r["mutants"]:
            by_op[m["op"]][m["verdict"]] += 1
    mutant_errors = collections.Counter(re.sub(r"gate\.v:\d+", "gate.v:N", m["error"] or "")[:120]
                                        for r in results for m in r["mutants"]
                                        if m["verdict"] in ("mutant error", "error"))
    verdicts = collections.Counter(m["verdict"] for r in results for m in r["mutants"])
    stats = {
        "yosys": version, "args": {k: v for k, v in vars(args).items()},
        "pool": dict(pool),
        "designs_sent": len(results),
        "parse": dict(parse),
        "parse_rate": parse["ok"] / max(1, len(results)),
        "parse_errors_top": parse_errors.most_common(10),
        "parse_seconds_median": percentile([r["parse_s"] for r in results], 0.5),
        "mutants_checked": len(secs),
        "verdicts": dict(verdicts),
        "differs_rate": verdicts["differs"] / max(1, len(secs)),
        "check_seconds": {"median": percentile(secs, 0.5), "p90": percentile(secs, 0.9),
                          "p99": percentile(secs, 0.99), "max": max(secs) if secs else None,
                          "total": round(sum(secs), 1)},
        "wall_minutes": round(wall / 60, 2), "jobs": args.jobs,
        "designs_with_site": {op: available[op] for op in OPERATORS},
        "by_operator": {op: dict(by_op[op]) for op in OPERATORS},
        "mutant_errors_top": mutant_errors.most_common(10),
        "examples": {name: {"examples": len(exs), "kinds": dict(collections.Counter(e["kind"] for e in exs)),
                            "groups": len({e["group"] for e in exs}),
                            "tokens_median": percentile(lengths[name], 0.5),
                            "tokens_max": max(lengths[name]) if lengths[name] else 0,
                            "blocks_median": percentile([len(e["state"]["blocks"]) for e in exs], 0.5)}
                     for name, exs in splits.items()},
        "dropped": dict(dropped),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nparse: {parse['ok']}/{len(results)} ok ({stats['parse_rate']:.1%}); {dict(parse)}")
    for e, n in parse_errors.most_common(5):
        print(f"   {n:4d}  {e}")
    print(f"mutants: {len(secs)} checked; {dict(verdicts)}; behavior changed {stats['differs_rate']:.1%}")
    cs = stats["check_seconds"]
    print(f"seconds per check: median {cs['median']}, p90 {cs['p90']}, p99 {cs['p99']}, max {cs['max']}; "
          f"{cs['total']} s in all, {stats['wall_minutes']} min wall with {args.jobs} jobs")
    print(f"{'operator':16s} {'designs':>7s} {'checked':>7s} {'differs':>7s} {'equiv':>6s} {'limit':>6s} "
          f"{'tmout':>6s} {'other':>6s}")
    for op in OPERATORS:
        c = by_op[op]
        print(f"{op:16s} {available[op]:7d} {sum(c.values()):7d} {c['differs']:7d} {c['equivalent']:6d} "
              f"{c['step limit']:6d} {c['timeout'] + c['sat timeout']:6d} "
              f"{sum(c.values()) - c['differs'] - c['equivalent'] - c['step limit'] - c['timeout'] - c['sat timeout']:6d}")
    for e, n in mutant_errors.most_common(5):
        print(f"   {n:4d}  {e}")
    for name, s in stats["examples"].items():
        print(f"{name}: {s}")
    print(f"done: {args.out} (train.jsonl, val.jsonl, checks.jsonl, stats.json)")


if __name__ == "__main__":
    main()
