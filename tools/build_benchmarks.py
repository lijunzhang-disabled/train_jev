#!/usr/bin/env python3
"""Convert the two real-bug benchmarks into one JSONL of records.

Run tools/fetch_benchmarks.sh first. Stdlib only, so it runs on the server as-is:

    python3 -I tools/build_benchmarks.py [--workers 16] [--offline]

Inputs
    data/raw/verification-benchmarks/        SVA security properties + buggy designs
    data/raw/hwe-bench/hwe_bench_full.jsonl  real RTL bug fixes

Outputs (all under data/benchmarks/)
    benchmarks.jsonl     one record per property (verification-benchmarks) or bug fix (hwe-bench)
    hwe_files/<org>__<repo>/<base sha>/<path>        RTL files before the fix (downloaded)
    hwe_files_fixed/<org>__<repo>/pr-<number>/<path> the same files with that PR's fix applied
    test_exclusions.txt  files that test records touch; keep them out of training data
    summary.json         counts and sanity checks

Record fields
    id           "vb/<design>/<property>" or "hwe/<org>__<repo>/pr-<number>"
    source       "verification-benchmarks" | "hwe-bench"
    design       e.g. "hackatdac21", "lowRISC/opentitan"
    split        "test" | "train"   (see HWE_TEST_REPOS / HWE_TEST_SINCE below)
    bug          {description, category, cwe: ["CWE-1234", ...]}
    property     {name, kind, text, fpv_result: "violation" | "no_violation"}  (verification-benchmarks)
                 or null (hwe-bench)
    rtl          [{file, module, buggy, fixed, changed_lines, autogen}]
                   file           path inside the design repo
                   buggy / fixed  local paths relative to the project root (fixed: hwe-bench only)
                   changed_lines  [[first, last], ...] lines of the buggy file the fix touches
                                  (hwe-bench only; pure insertions span the two lines around them)
    problem_statement, fix_patch, created_at   (hwe-bench only, else null)
"""
import argparse
import collections
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "benchmarks")

VB_DIR = os.path.join(RAW, "verification-benchmarks")
VB_DESIGNS = {  # design folder -> RTL root inside it
    "hackatdac18": "hackatdac18",
    "hackatdac19": "hackatdac19/design",
    "hackatdac21": "hackatdac21/design",
    "or1200": "or1200/buggy-or1200",
}
HWE_JSONL = os.path.join(RAW, "hwe-bench", "hwe_bench_full.jsonl")
HWE_REPOS = {"opentitan", "ibex", "cva6", "caliptra-rtl"}  # SystemVerilog; Chisel repos are out of scope
# Proposed split (open question in PLAN.md): these repos, plus OpenTitan from this date on, are test-only.
HWE_TEST_REPOS = {"ibex", "cva6", "caliptra-rtl"}
HWE_TEST_SINCE = "2024-01-01"

HDL_EXT = (".sv", ".svh", ".v", ".vh")
NON_RTL_DIR = re.compile(r"(^|/)(dv|tb|test|tests|verif|fpv|formal|sim)(/|$)")


def rel(path):
    return os.path.relpath(path, ROOT)


# ---------------------------------------------------------------- verification-benchmarks

def md_table(path):
    """Rows of the first markdown table in a file, as lists of unescaped cell strings."""
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.startswith("|") or re.match(r"^\|\s*-", line):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:]]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        rows.append([unescape_md(c) for c in cells])
    return rows[0], rows[1:]


def unescape_md(s):
    s = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|~<>])", r"\1", s)
    return s.replace("’", "'")  # a curly quote in one property (8’h59) is meant as an SV literal


def parse_property(cell):
    m = re.match(r"(assert|cover|assume)\s+-name\s+(\S+)\s*(\{.*\})\s*$", cell, re.S)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "assert", None, cell


def fpv_result(cell):
    c = cell.lower()
    if "no violation" in c:
        return "no_violation"
    if "violation" in c:
        return "violation"
    raise ValueError(f"unknown FPV result: {cell!r}")


def cwe_list(cell):
    return [f"CWE-{n}" for n in re.findall(r"\d+", cell)]


def module_index(rtl_root):
    """module name -> files defining it, plus file stem -> files, over all HDL files."""
    by_module, by_stem = collections.defaultdict(list), collections.defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(rtl_root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in filenames:
            if not f.endswith(HDL_EXT):
                continue
            p = os.path.join(dirpath, f)
            by_stem[f.split(".")[0]].append(p)
            with open(p, encoding="utf-8", errors="replace") as fh:
                for m in re.finditer(r"^\s*module\s+(\w+)", fh.read(), re.M):
                    by_module[m.group(1)].append(p)
    return by_module, by_stem


def files_by_signal_names(rtl_root, text):
    """Fallback for properties that name bare signals: files containing the most of their identifiers."""
    words = set(re.findall(r"\b[A-Za-z_]\w*\b", text)) - {"rst", "h", "b", "d"}
    best, best_n = [], 0
    for dirpath, dirnames, filenames in os.walk(rtl_root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in filenames:
            if f.endswith(HDL_EXT):
                p = os.path.join(dirpath, f)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    found = words & set(re.findall(r"\b[A-Za-z_]\w*\b", fh.read()))
                if len(found) > best_n:
                    best, best_n = [p], len(found)
                elif len(found) == best_n and best_n:
                    best.append(p)
    return best


def vb_records():
    records = []
    for design, sub in VB_DESIGNS.items():
        rtl_root = os.path.join(VB_DIR, sub)
        by_module, by_stem = module_index(rtl_root)
        header, rows = md_table(os.path.join(VB_DIR, design, "properties.md"))
        col = {h: i for i, h in enumerate(header)}
        for row in rows:
            kind, name, text = parse_property(row[col["Assertion"]])
            name = name or row[col["ID"]]
            if "Module" in col:
                modules = [row[col["Module"]]]
            else:  # OR1200: take the modules the assertion refers to, most-referenced first
                refs = collections.Counter(re.findall(r"\b(or1200_\w+?)\s*\.", text))
                modules = [m for m, _ in refs.most_common()]
            found = [(mod, f) for mod in modules for f in (by_module.get(mod) or by_stem.get(mod) or [])]
            if not found:
                found = [(None, f) for f in files_by_signal_names(rtl_root, text)]
            rtl = [{"file": os.path.relpath(f, rtl_root), "module": mod, "buggy": rel(f),
                    "fixed": None, "changed_lines": None, "autogen": False} for mod, f in found]
            records.append({
                "id": f"vb/{design}/{name}",
                "source": "verification-benchmarks",
                "design": design,
                "split": "test",
                "bug": {
                    "description": row[col["Bug Description"]] if "Bug Description" in col else None,
                    "category": row[col["Type"]] if "Type" in col else None,
                    "cwe": cwe_list(row[col["CWE-ID"]]),
                },
                "property": {"name": name, "kind": kind, "text": text,
                             "fpv_result": fpv_result(row[col["FPV Verification Results"]])},
                "rtl": rtl,
                "problem_statement": None, "fix_patch": None, "created_at": None,
            })
    return records


# ---------------------------------------------------------------- hwe-bench

def split_patch(patch):
    """[(path, is_new, is_deleted, hunks)]; hunk = (old_start, old_len, [(tag, text), ...])."""
    files = []
    for block in re.split(r"^diff --git ", patch, flags=re.M)[1:]:
        path = re.match(r"a/(\S+) b/(\S+)", block).group(2)
        head = block.split("\n@@", 1)[0]
        is_new = "\nnew file mode" in head or "\n--- /dev/null" in head
        is_deleted = "\ndeleted file mode" in head or "\n+++ /dev/null" in head
        hunks, cur = [], None
        for line in block.split("\n"):
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                cur = (int(m.group(1)), int(m.group(2) or 1), [])
                hunks.append(cur)
            elif cur is not None and line[:1] in (" ", "-", "+"):
                cur[2].append((line[0], line[1:]))
            elif cur is not None and line == "":
                cur[2].append((" ", ""))  # some patches drop the space on empty context lines
        files.append((path, is_new, is_deleted, hunks))
    return files


def find_hunk(key, block, expected, lo):
    """Start index of `block` in `key` at or after `lo`, closest to `expected` (like patch's offset search)."""
    if not block:
        return max(expected, lo)
    first, n = block[0], len(block)
    hits = [k for k in range(lo, len(key) - n + 1) if key[k] == first and key[k:k + n] == block]
    return min(hits, key=lambda k: abs(k - expected)) if hits else None


def trim_context(lines, fuzz):
    """Drop up to `fuzz` context lines from each end of a hunk (patch's fuzz factor)."""
    lead = 0
    while lead < fuzz and lead < len(lines) and lines[lead][0] == " ":
        lead += 1
    end = len(lines)
    while end > lead and len(lines) - end < fuzz and lines[end - 1][0] == " ":
        end -= 1
    return lead, lines[lead:end]


def apply_hunks(old_text, hunks):
    """Apply one file's hunks, tolerating line offsets, up to 2 lines of fuzz and CRLF files.

    Returns (new_text, changed_line_ranges, max_offset) or raises ValueError.
    """
    old = old_text.split("\n")
    key = [line.rstrip("\r") for line in old]  # compare without CR so CRLF files match LF patches
    eol = "\r" if sum(line.endswith("\r") for line in old) > len(old) // 2 else ""
    out, pos, ranges, max_offset = [], 0, [], 0
    for old_start, old_len, lines in hunks:
        # trailing empty context lines may come from the split on "\n" at the end of the block
        while lines and lines[-1] == (" ", "") and sum(t != "+" for t, _ in lines) > old_len:
            lines = lines[:-1]
        expected = old_start - 1 if old_len > 0 else old_start
        for fuzz in (0, 1, 2):
            lead, body = trim_context(lines, fuzz)
            start = find_hunk(key, [t.rstrip("\r") for tag, t in body if tag != "+"], expected + lead, pos)
            if start is not None:
                break
        else:
            raise ValueError(f"hunk near line {expected + 1} not found")
        max_offset = max(max_offset, abs(start - expected - lead))
        out.extend(old[pos:start])
        pos = start
        run = None
        for tag, text in body:
            if tag == " ":
                out.append(old[pos])
                pos += 1
                if run:
                    ranges.append(run)
                    run = None
            elif tag == "-":
                run = [min(run[0], pos + 1), max(run[1], pos + 1)] if run else [pos + 1, pos + 1]
                pos += 1
            else:  # "+": replaces the removed lines of this run, or is a pure insertion between pos and pos + 1
                if run is None:
                    run = [max(pos, 1), min(pos + 1, len(old))]
                out.append(text.rstrip("\r") + eol)
        if run:
            ranges.append(run)
    out.extend(old[pos:])
    return "\n".join(out), ranges, max_offset


def fetch(url, dest, retries=3):
    if os.path.exists(dest):
        return True
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest + ".part", "wb") as fh:
                fh.write(data)
            os.replace(dest + ".part", dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    return False


def hwe_records(workers, offline):
    cases = [json.loads(line) for line in open(HWE_JSONL, encoding="utf-8")]
    cases = [c for c in cases if c["repo"] in HWE_REPOS and c["level1"] == "RTL_BUG_FIX"]

    jobs = {}  # local buggy path -> url
    plan = []
    for c in cases:
        sha, slug = c["base"]["sha"], f'{c["org"]}__{c["repo"]}'
        files = []
        for path, is_new, is_deleted, hunks in split_patch(c["fix_patch"]):
            if not path.endswith(HDL_EXT) or NON_RTL_DIR.search(path) or is_new or is_deleted:
                continue
            buggy = os.path.join(OUT, "hwe_files", slug, sha, path)
            fixed = os.path.join(OUT, "hwe_files_fixed", slug, f'pr-{c["number"]}', path)  # per PR: PRs can share a base
            jobs[buggy] = f'https://raw.githubusercontent.com/{c["org"]}/{c["repo"]}/{sha}/{path}'
            files.append((path, hunks, buggy, fixed))
        plan.append((c, files))

    missing = [p for p in jobs if not os.path.exists(p)]
    print(f"hwe-bench: {len(cases)} SystemVerilog bug fixes, {len(jobs)} RTL files, {len(missing)} to download")
    failed = set()
    if missing and not offline:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for path, ok in zip(missing, pool.map(lambda p: fetch(jobs[p], p), missing)):
                if not ok:
                    failed.add(path)
    failed |= {p for p in jobs if not os.path.exists(p)}

    records, apply_errors, test_touched, offsets = [], [], set(), []
    for c, files in plan:
        rtl = []
        for path, hunks, buggy, fixed in files:
            entry = {"file": path, "module": None, "buggy": rel(buggy), "fixed": None,
                     "changed_lines": None, "autogen": "/ip_autogen/" in path or "/autogen/" in path}
            if buggy not in failed:
                with open(buggy, encoding="utf-8", errors="surrogateescape", newline="") as fh:
                    old_text = fh.read()
                try:
                    new_text, ranges, offset = apply_hunks(old_text, hunks)
                    if offset:
                        offsets.append(offset)
                    os.makedirs(os.path.dirname(fixed), exist_ok=True)
                    with open(fixed, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
                        fh.write(new_text)
                    entry["fixed"], entry["changed_lines"] = rel(fixed), ranges
                except ValueError as e:
                    apply_errors.append(f'{c["org"]}/{c["repo"]}#{c["number"]} {path}: {e}')
            rtl.append(entry)
        test = c["repo"] in HWE_TEST_REPOS or c["created_at"][:10] >= HWE_TEST_SINCE
        if test:
            for patch in (c["fix_patch"], c.get("test_patch") or ""):
                test_touched.update((f'{c["org"]}/{c["repo"]}', p) for p, *_ in split_patch(patch))
        records.append({
            "id": f'hwe/{c["org"]}__{c["repo"]}/pr-{c["number"]}',
            "source": "hwe-bench",
            "design": f'{c["org"]}/{c["repo"]}',
            "split": "test" if test else "train",
            "bug": {"description": c["title"], "category": c["level2"], "cwe": []},
            "property": None,
            "rtl": rtl,
            "problem_statement": c["problem_statement"],
            "fix_patch": c["fix_patch"],
            "created_at": c["created_at"],
        })
    return records, sorted(failed), apply_errors, test_touched, offsets


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads")
    ap.add_argument("--offline", action="store_true", help="don't download; use cached files only")
    args = ap.parse_args()
    for p in (VB_DIR, HWE_JSONL):
        if not os.path.exists(p):
            sys.exit(f"missing {rel(p)}: run tools/fetch_benchmarks.sh first")
    os.makedirs(OUT, exist_ok=True)

    vb = vb_records()
    hwe, failed, apply_errors, touched, offsets = hwe_records(args.workers, args.offline)
    records = vb + hwe

    with open(os.path.join(OUT, "benchmarks.jsonl"), "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT, "test_exclusions.txt"), "w", encoding="utf-8") as fh:
        fh.write("# Keep these out of training data.\n")
        fh.write("# Whole design families used by verification-benchmarks:\n")
        for fam in ("PULPissimo / RI5CY (hackatdac18)", "Ariane / CVA6 (hackatdac19, hackatdac21)",
                    "OpenPiton (hackatdac21)", "OR1200"):
            fh.write(f"family\t{fam}\n")
        fh.write("# Files touched by hwe-bench test records (fix and test patches):\n")
        for design, path in sorted(touched):
            fh.write(f"file\t{design}\t{path}\n")

    def count(rs, key):
        return dict(collections.Counter(key(r) for r in rs).most_common())
    summary = {
        "records": len(records),
        "by_source_split": count(records, lambda r: f'{r["source"]}/{r["split"]}'),
        "vb_by_design": count(vb, lambda r: r["design"]),
        "vb_fpv_result": count(vb, lambda r: r["property"]["fpv_result"]),
        "vb_property_kind": count(vb, lambda r: r["property"]["kind"]),
        "vb_cwe": count([c for r in vb for c in r["bug"]["cwe"]], lambda c: c),
        "vb_without_cwe": sum(not r["bug"]["cwe"] for r in vb),
        "vb_without_rtl_file": [r["id"] for r in vb if not r["rtl"]],
        "hwe_by_repo_split": count(hwe, lambda r: f'{r["design"]}/{r["split"]}'),
        "hwe_by_category": count(hwe, lambda r: r["bug"]["category"]),
        "hwe_rtl_files": sum(len(r["rtl"]) for r in hwe),
        "hwe_rtl_files_with_fix_applied": sum(1 for r in hwe for f in r["rtl"] if f["fixed"]),
        "hwe_files_applied_at_offset": {"count": len(offsets), "max_lines": max(offsets, default=0)},
        "hwe_records_without_rtl": [r["id"] for r in hwe if not r["rtl"]],
        "hwe_download_failures": [rel(p) for p in failed],
        "hwe_apply_errors": apply_errors,
        "test_exclusion_files": len(touched),
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list) or len(v) < 8}, indent=2))
    for k in ("hwe_download_failures", "hwe_apply_errors"):
        if len(summary[k]) >= 8:
            print(f"{k}: {len(summary[k])} (see summary.json)")


if __name__ == "__main__":
    main()
