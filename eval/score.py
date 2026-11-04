import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mathruler.grader import grade_answer

from infer import BENCHMARK_FILES, make_sample_uid, needs_answer

LABELS = {
    "vstar": "V*",
    "zoombench": "ZoomBench",
    "hrbench-4k": "HR-Bench 4K",
    "hrbench-8k": "HR-Bench 8K",
    "mme-realworld": "MME-RealWorld EN",
    "mme-realworld-cn": "MME-RealWorld CN",
}
MCQ = {"vstar", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn"}
PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)
JUDGE_MAX_TOKENS = 2048
JUDGE_CONTEXT = 131072
JUDGE_RESERVE = 128
JUDGE_WORKERS_PER_REPLICA = 64
UNRESOLVED = f"unresolved_ctx{JUDGE_CONTEXT}_as_no"

FULLWIDTH = str.maketrans({"（": "(", "）": ")", "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E"})
FINAL = re.compile(
    r"(?i:final answer|correct answer|correct option|best answer is|the answer is|answer is|answer\s*:|"
    r"答案(?:是|为)?|最佳答案为|正确答案(?:是|为)?)"
    r"[^A-Za-z0-9一-鿿]{0,12}?([A-E])(?![A-Za-z])")
PAREN = re.compile(r"\(([A-E])\)")
LEAD = re.compile(r"^[\s\*\(\[]*([A-E])(?:[\s\.\)\]:\*]|$)")


def extract_first_option(text):
    if not text:
        return ""
    for pattern in (r"\(([A-Z])\)", r"([A-Z])[\.\)\s]", r"([A-Z])"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    match = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))", answer.strip())
    return match.group(1) if match else ""


def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def extract_answer(raw):
    if "<answer>" in raw:
        start = raw.find("<answer>")
        end = raw.find("</answer>")
        if start != -1 and end != -1:
            return raw[start + len("<answer>"):end].strip()
    if "Answer:" in raw:
        return raw[raw.find("Answer:"):].strip()
    return raw.strip()


def parse_choice(text):
    match = None
    for match in FINAL.finditer(text[-800:]):
        pass
    if match:
        return match.group(1)
    stripped = text.strip()
    lead = LEAD.match(stripped)
    if lead and len(stripped) <= 300:
        return lead.group(1)
    tail = PAREN.findall(text[-300:])
    if tail:
        return tail[-1]
    return lead.group(1) if lead else None


def rule_source(benchmark, gt, extracted):
    try:
        if grade_answer(gt, extracted):
            return "mathruler"
    except Exception:
        pass
    if benchmark in MCQ:
        try:
            if first_letter_match(gt, extracted):
                return "first letter"
        except Exception:
            pass
    return None


def decide(benchmark, row):
    official = str(row.get("judge", "")).strip().lower() == "yes"
    correct = official
    if official and row.get("judge_source") == "first letter" and benchmark in MCQ:
        choice = parse_choice(str(row.get("model_answer") or "").translate(FULLWIDTH))
        row["final_choice"] = choice
        if choice is not None and choice != str(row.get("response", "")).strip():
            correct = False
    return int(official), int(correct)


def is_verdict(text):
    return str(text or "").strip().lower() in ("yes", "no")


def record_key(row):
    payload = json.dumps([row.get("sample_uid"), row.get("query"), row.get("model_answer")], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PromptFitter:
    def __init__(self, tokenizer_name):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.limit = JUDGE_CONTEXT - JUDGE_MAX_TOKENS - JUDGE_RESERVE

    def count(self, prompt):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokens = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=False)
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], (list, tuple)):
            tokens = tokens[0]
        return len(tokens)

    def __call__(self, question, gt, response):
        def render(value):
            return PROMPT_TEMPLATE.format(question=question, gt=gt, response=value)

        prompt = render(response)
        if len(prompt.encode("utf-8")) + 1024 <= self.limit or self.count(prompt) <= self.limit:
            return prompt, False
        if self.count(render("")) > self.limit:
            raise ValueError("judge question and reference exceed the judge context")
        lo, hi = 0, len(response)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(render(response[:mid])) <= self.limit:
                lo = mid
            else:
                hi = mid - 1
        return render(response[:lo]), True


class Judge:
    def __init__(self, bases, model):
        from openai import OpenAI

        self.OpenAI = OpenAI
        self.bases = bases
        self.model = model
        self.local = threading.local()
        self.counter = itertools.count()
        self.lock = threading.Lock()

    def client(self):
        client = getattr(self.local, "client", None)
        if client is None:
            with self.lock:
                base = self.bases[next(self.counter) % len(self.bases)]
            client = self.OpenAI(api_key="EMPTY", base_url=base, timeout=600)
            self.local.client = client
        return client

    def __call__(self, prompt):
        client = self.client()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=JUDGE_MAX_TOKENS,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                permanent = getattr(exc, "status_code", None) in (400, 401, 403, 404, 413, 422)
                if permanent or attempt == 2:
                    print(f"[judge] request failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    break
                time.sleep(1.0)
        return "[API_ERROR] judge request failed"


class Journal:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        cache = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if is_verdict(entry.get("judge")):
                        cache[entry.pop("key")] = entry
        return cache

    def append(self, row):
        entry = {"key": record_key(row)}
        entry.update({k: row[k] for k in ("judge", "judge_source", "judge_raw") if k in row})
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_cache(judge_dir, benchmarks, journal):
    cache = {}
    for benchmark in benchmarks:
        path = judge_dir / f"{benchmark}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            source = str(row.get("judge_source", ""))
            if source.startswith(("llm", "unresolved")) and is_verdict(row.get("judge")):
                cache[record_key(row)] = {k: row[k] for k in ("judge", "judge_source", "judge_raw") if k in row}
    cache.update(journal.load())
    return cache


def load_answers(run_dir, data_dir, benchmark):
    path = run_dir / "answers" / f"{benchmark}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run infer.py first")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(Path(data_dir) / BENCHMARK_FILES[benchmark], encoding="utf-8") as f:
        expected = {make_sample_uid(item, benchmark) for item in json.load(f)}
    uids = [row.get("sample_uid") or make_sample_uid(row, benchmark) for row in rows]
    if len(uids) != len(set(uids)) or set(uids) != expected or any(needs_answer(row) for row in rows):
        raise SystemExit(f"{path} is incomplete or has failed answers; rerun infer.py")
    return rows


def run_judge(pending, judge, journal):
    def call(index):
        return index, judge(pending[index][1])

    def record(index, text):
        row = pending[index][0]
        row["judge"], row["judge_source"] = text, "llm"
        if is_verdict(text):
            journal.append(row)

    workers = JUDGE_WORKERS_PER_REPLICA * len(judge.bases)
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, i) for i in range(len(pending))]
        for future in tqdm(as_completed(futures), total=len(futures), desc="judge", dynamic_ncols=True):
            record(*future.result())
        retry = [i for i, (row, _) in enumerate(pending)
                 if not is_verdict(row["judge"]) and not str(row["judge"]).startswith("[API_ERROR]")]
        for future in as_completed([pool.submit(call, i) for i in retry]):
            record(*future.result())
    for i in retry:
        row = pending[i][0]
        if not is_verdict(row["judge"]):
            row.update(judge_raw=row["judge"], judge="No", judge_source=UNRESOLVED)
            journal.append(row)
    return sum(str(row["judge"]).startswith("[API_ERROR]") for row, _ in pending)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_FILES))
    parser.add_argument("--judge-bases", default="")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-tokenizer", default="openai/gpt-oss-120b")
    parser.add_argument("--name", default=None)
    parser.add_argument("--count-pending", action="store_true")
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = [b for b in benchmarks if b not in BENCHMARK_FILES]
    if unknown:
        raise SystemExit(f"unknown benchmarks: {', '.join(unknown)}")
    run_dir = Path(args.run_dir)
    judge_dir = run_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(judge_dir / "journal.jsonl")
    cache = load_cache(judge_dir, benchmarks, journal)
    fitter = PromptFitter(args.judge_tokenizer)

    tables, pending = {}, []
    for benchmark in benchmarks:
        rows = load_answers(run_dir, args.data_dir, benchmark)
        for row in rows:
            for key in ("judge", "judge_source", "judge_raw", "judge_input", "judge_truncated", "final_choice",
                        "correct", "correct_official"):
                row.pop(key, None)
            extracted = extract_answer(row["model_answer"])
            row["extracted_answer"] = extracted
            source = rule_source(benchmark, row["response"], extracted)
            if source:
                row["judge"], row["judge_source"] = "Yes", source
                continue
            cached = cache.get(record_key(row))
            if cached:
                row.update(cached)
                continue
            prompt, truncated = fitter(row["query"].replace("<image>", ""), row["response"], extracted)
            if truncated:
                row["judge_truncated"] = True
            pending.append((row, prompt))
        tables[benchmark] = rows

    if args.count_pending:
        print(len(pending))
        return
    failures = 0
    if pending:
        bases = [u.strip() for u in args.judge_bases.split(",") if u.strip()]
        if not bases:
            raise SystemExit(f"{len(pending)} responses need the LLM judge; pass --judge-bases")
        print(f"[judge] {len(pending)} responses go to the LLM judge", flush=True)
        failures = run_judge(pending, Judge(bases, args.judge_model), journal)

    results = {"name": args.name or run_dir.name, "benchmarks": {}}
    for benchmark, rows in tables.items():
        n = len(rows)
        official = corrected = 0
        for row in rows:
            row["correct_official"], row["correct"] = decide(benchmark, row)
            official += row["correct_official"]
            corrected += row["correct"]
        results["benchmarks"][benchmark] = {"n": n, "corrected": 100.0 * corrected / n,
                                            "official": 100.0 * official / n}
        tmp = judge_dir / f"{benchmark}.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        os.replace(tmp, judge_dir / f"{benchmark}.json")
    if failures:
        raise SystemExit(f"{failures} judge requests failed; rerun to resume")

    scores = results["benchmarks"].values()
    results["mean_corrected"] = sum(s["corrected"] for s in scores) / len(benchmarks)
    results["mean_official"] = sum(s["official"] for s in scores) / len(benchmarks)
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if journal.path.exists():
        journal.path.unlink()

    print(f"\n{results['name']}")
    print(f"{'benchmark':<18}{'n':>7}{'corrected':>11}{'official':>10}")
    for benchmark, s in results["benchmarks"].items():
        print(f"{LABELS[benchmark]:<18}{s['n']:>7}{s['corrected']:>11.2f}{s['official']:>10.2f}")
    print(f"{f'average ({len(benchmarks)})':<18}{'':>7}{results['mean_corrected']:>11.2f}{results['mean_official']:>10.2f}")
    print(f"results: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
