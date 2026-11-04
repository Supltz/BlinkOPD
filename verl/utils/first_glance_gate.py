import re

import numpy as np

_LAST_LINE = re.compile(r"^\(?([A-H])(?:[\.\):]|\s*$)")
_FINAL = re.compile(
    r"(?i:final answer|correct answer|correct option|correct choice|best answer|best choice|best option|"
    r"the answer is|answer is|answer would be|answer should be|answer\s*:)"
    r"[^A-Za-z0-9]{0,12}?([A-H])(?![A-Za-z])")
_OPTION = re.compile(r"(?i:\boption)\s*\(?([A-H])\)?(?![A-Za-z])")
_PAREN = re.compile(r"\(([A-H])\)")
_LEAD = re.compile(r"^[\s\*\(\[]*([A-H])(?:[\.\)\]:\*]|\s*$)")


def parse_answer_letter(text, options):
    if not text:
        return None
    allowed = set(options)
    lines = [ln.strip().strip("*").strip().strip('"').strip() for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln]
    if lines:
        m = _LAST_LINE.match(lines[-1].replace("**", ""))
        if m and m.group(1) in allowed:
            return m.group(1)
    for pattern, span in ((_FINAL, 800), (_OPTION, 400), (_PAREN, 300)):
        hits = [h for h in pattern.findall(text[-span:]) if h in allowed]
        if hits:
            return hits[-1]
    if len(text.strip()) <= 300:
        m = _LEAD.match(text.strip())
        if m and m.group(1) in allowed:
            return m.group(1)
    return None


def conflict_gate(response_texts, first_letters, option_strs):
    n = len(response_texts)
    if not (len(first_letters) == len(option_strs) == n):
        raise ValueError("responses, first glances and options must have the same length")
    parsed = [parse_answer_letter(t, o) for t, o in zip(response_texts, option_strs)]
    gate = np.array([1.0 if (p is None or p != f) else 0.0 for p, f in zip(parsed, first_letters)], dtype=np.float32)
    stats = {
        "conflict_rate": float(gate.mean()) if n else 0.0,
        "parse_fail_rate": float(np.mean([p is None for p in parsed])) if n else 0.0,
    }
    return gate, stats


def gate_to_scale(gate, on_scale, off_scale):
    gate = np.asarray(gate, dtype=np.float32)
    return (off_scale + (on_scale - off_scale) * gate).astype(np.float32)
