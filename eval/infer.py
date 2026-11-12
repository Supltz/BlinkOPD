import argparse
import base64
import hashlib
import json
import mimetypes
import multiprocessing
import os
import queue
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

BENCHMARK_FILES = {
    "vstar": "vstar.json",
    "zoombench": "zoombench.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
}
MAX_TOKENS = 32768
MAX_RETRIES = 3
WORKERS_PER_REPLICA = 64
VSTAR_MAX_IMAGE_SIZE = 20 * 1024 * 1024


def make_sample_uid(item, benchmark):
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable = {"benchmark": benchmark, "images": item.get("images") or [], "query": item.get("query", "")}
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def needs_answer(record):
    answer = record.get("model_answer")
    if not isinstance(answer, str) or not answer.strip():
        return True
    answer = answer.strip()
    return answer.startswith("[API_ERROR]") or answer.startswith("[FUTURE_ERROR]")


def read_answers(path, benchmark):
    order, records, changed = [], {}, False
    if not path.exists():
        return order, records, changed
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            try:
                record = json.loads(line) if line else None
            except ValueError:
                record = None
            if record is None:
                changed = True
                continue
            uid = record.get("sample_uid") or make_sample_uid(record, benchmark)
            record["sample_uid"] = uid
            if uid not in records:
                order.append(uid)
                records[uid] = record
                continue
            changed = True
            if needs_answer(records[uid]) or not needs_answer(record):
                records[uid] = record
    return order, records, changed


def write_answers(path, order, records):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for uid in order:
            f.write(json.dumps(records[uid], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def image_path(item, data_dir):
    path = Path((item.get("images") or [""])[0])
    return path if path.is_absolute() else Path(data_dir) / path


def image_to_data_uri(path, benchmark):
    if benchmark == "vstar":
        img = Image.open(path).convert("RGB")
        output = BytesIO()
        img.save(output, format="PNG")
        data = output.getvalue()
        while len(data) > VSTAR_MAX_IMAGE_SIZE and img.size[0] > 100 and img.size[1] > 100:
            img = img.resize((int(img.size[0] * 0.75), int(img.size[1] * 0.75)), Image.Resampling.LANCZOS)
            output = BytesIO()
            img.save(output, format="PNG")
            data = output.getvalue()
        return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_request(item, benchmark, data_dir, model_id):
    query = item.get("query", "").replace("<image>", "").strip()
    uri = image_to_data_uri(image_path(item, data_dir), benchmark)
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                             {"type": "text", "text": query}]}]
    return {
        "model": model_id,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def answer(client, item, benchmark, data_dir, model_id):
    request = build_request(item, benchmark, data_dir, model_id)
    model_answer = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**request)
            model_answer = (resp.choices[0].message.content or "").strip()
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                model_answer = f"[API_ERROR] {exc}"
            else:
                time.sleep(1.0)
    return dict(item, model_answer=model_answer)


def client_process(endpoint, threads, data_dir, model_id, work, results):
    from openai import OpenAI

    def serve():
        client = OpenAI(api_key="EMPTY", base_url=endpoint, timeout=3600)
        while True:
            task = work.get()
            if task is None:
                return
            benchmark, item = task
            try:
                record = answer(client, item, benchmark, data_dir, model_id)
            except Exception as exc:
                record = dict(item, model_answer=f"[FUTURE_ERROR] {exc}")
            results.put((benchmark, record))

    pool = [threading.Thread(target=serve, daemon=True) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()


def plan(benchmarks, data_dir, run_dir):
    jobs = {}
    for benchmark in benchmarks:
        source = Path(data_dir) / BENCHMARK_FILES[benchmark]
        if not source.exists():
            raise SystemExit(f"missing {source}; run prepare_data.py first")
        with open(source, encoding="utf-8") as f:
            items = json.load(f)
        out_path = Path(run_dir) / "answers" / f"{benchmark}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        order, records, changed = read_answers(out_path, benchmark)
        if changed:
            write_answers(out_path, order, records)
        done = {uid for uid in order if not needs_answer(records[uid])}
        todo = []
        for item in items:
            uid = make_sample_uid(item, benchmark)
            if uid not in done:
                todo.append(dict(item, sample_uid=uid))
        expected = {make_sample_uid(item, benchmark) for item in items}
        jobs[benchmark] = (out_path, todo, expected)
    return jobs


def file_size(item, data_dir):
    try:
        return os.path.getsize(image_path(item, data_dir))
    except (OSError, TypeError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_FILES))
    parser.add_argument("--api-bases", default="")
    parser.add_argument("--model-id", default="eval-model")
    parser.add_argument("--count-pending", action="store_true")
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = [b for b in benchmarks if b not in BENCHMARK_FILES]
    if unknown:
        raise SystemExit(f"unknown benchmarks: {', '.join(unknown)}")
    data_dir = str(Path(args.data_dir).resolve())
    jobs = plan(benchmarks, data_dir, args.run_dir)
    work_items = [(b, item) for b in benchmarks for item in jobs[b][1]]
    if args.count_pending:
        print(len(work_items))
        return

    bases = [u.strip() for u in args.api_bases.split(",") if u.strip()]
    if work_items and not bases:
        raise SystemExit("--api-bases is required when answers are missing")
    if work_items:
        from tqdm import tqdm

        work_items.sort(key=lambda pair: -file_size(pair[1], data_dir))
        context = multiprocessing.get_context("spawn")
        work, results = context.Queue(), context.Queue()
        workers = [context.Process(target=client_process,
                                   args=(base, WORKERS_PER_REPLICA, data_dir, args.model_id, work, results),
                                   daemon=True) for base in bases]
        for worker in workers:
            worker.start()
        for pair in work_items:
            work.put(pair)
        for _ in range(len(bases) * WORKERS_PER_REPLICA):
            work.put(None)
        handles = {b: open(jobs[b][0], "a", encoding="utf-8") for b in benchmarks}
        received = 0
        try:
            with tqdm(total=len(work_items), desc="inference", unit="sample", dynamic_ncols=True) as bar:
                while received < len(work_items):
                    try:
                        benchmark, record = results.get(timeout=10)
                    except queue.Empty:
                        if not any(worker.is_alive() for worker in workers):
                            break
                        continue
                    handles[benchmark].write(json.dumps(record, ensure_ascii=False) + "\n")
                    handles[benchmark].flush()
                    received += 1
                    bar.update(1)
        finally:
            for handle in handles.values():
                handle.close()
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
            work.cancel_join_thread()

    problems = []
    for benchmark in benchmarks:
        out_path, _, expected = jobs[benchmark]
        order, records, changed = read_answers(out_path, benchmark)
        if changed:
            write_answers(out_path, order, records)
        failed = sum(needs_answer(records[uid]) for uid in order)
        missing = len(expected - set(order))
        print(f"[infer] {benchmark}: {len(order)} answers, {failed} failed, {missing} missing", flush=True)
        if failed or missing or set(order) != expected:
            problems.append(benchmark)
    if problems:
        print(f"[infer] incomplete: {', '.join(problems)}; rerun to resume", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
