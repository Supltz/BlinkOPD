import argparse
import base64
import json
import math
import mimetypes
import os
import queue
import sys
import threading
import time
from pathlib import Path

import scoring

BENCHMARKS = ['mmstar', 'cvbench', 'mmvp', 'pope', 'hallusionbench', 'mathvista', 'mathverse', 'mathvision', 'wemath', 'dynamath']
PERCEPTION = BENCHMARKS[:5]
REASONING = BENCHMARKS[5:]
TITLES = {'mmstar': 'MMStar', 'cvbench': 'CV-Bench', 'mmvp': 'MMVP', 'pope': 'POPE', 'hallusionbench': 'HallB',
          'mathvista': 'MathVista', 'mathverse': 'MathVerse', 'mathvision': 'MATH-V', 'wemath': 'WeMath', 'dynamath': 'DynaMath'}
MAIN_METRIC = {'hallusionbench': 'aAcc'}
JUDGED = ['mmstar', 'cvbench', 'mmvp', 'hallusionbench', 'mathvista', 'mathverse']
INFER_ORDER = ['dynamath', 'mathverse', 'mathvision', 'wemath', 'mathvista', 'mmstar', 'cvbench', 'hallusionbench', 'mmvp', 'pope']
MAX_TOKENS = 32768
MAX_RETRIES = 3
SERVED_MODEL = 'general-eval'


def parse_benchmarks(arg):
    benches = [b.strip() for b in arg.split(',') if b.strip()]
    unknown = [b for b in benches if b not in BENCHMARKS]
    if unknown:
        raise SystemExit(f'unknown benchmarks: {", ".join(unknown)}')
    return benches


def load_items(data_dir, bench):
    path = Path(data_dir) / f'{bench}.json'
    if not path.exists():
        raise SystemExit(f'missing {path}; run general_eval/prepare_data.py first')
    return json.loads(path.read_text(encoding='utf-8'))


def answers_path(run_dir, bench):
    return Path(run_dir) / 'answers' / f'{bench}.jsonl'


def valid_answer(a):
    return isinstance(a, str) and bool(a.strip()) and not a.startswith(('[API_ERROR]', '[FUTURE_ERROR]'))


def read_answers(run_dir, bench):
    p, ok = answers_path(run_dir, bench), {}
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if valid_answer(r.get('model_answer')):
                ok[r['question_id']] = r['model_answer']
    return ok


def image_data_uri(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    mime = mimetypes.guess_type(str(path))[0] or 'image/jpeg'
    return f'data:{mime};base64,{b64}'


def build_messages(item, data_dir):
    query = item.get('query', '').replace('<image>', '').strip()
    if item.get('images'):
        uri = image_data_uri(Path(data_dir) / item['images'][0])
        return [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': uri}}, {'type': 'text', 'text': query}]}]
    return [{'role': 'user', 'content': [{'type': 'text', 'text': query}]}]


def request_answer(client, item, data_dir):
    messages = build_messages(item, data_dir)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(model=SERVED_MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0,
                                               extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f'[API_ERROR] {e}'
            time.sleep(1.0)


def pending(data_dir, run_dir, benches):
    work = []
    for bench in [b for b in INFER_ORDER if b in benches]:
        done = read_answers(run_dir, bench)
        work += [(bench, it) for it in load_items(data_dir, bench) if it['question_id'] not in done]
    return work


def raise_nofile():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except (ValueError, OSError):
            pass


def run_round(work, urls, threads, data_dir, run_dir):
    from openai import OpenAI
    q = queue.Queue()
    for w in work:
        q.put(w)
    locks = {b: threading.Lock() for b in BENCHMARKS}
    stats = {'done': 0, 'errors': 0}
    stat_lock = threading.Lock()
    answers_path(run_dir, 'x').parent.mkdir(parents=True, exist_ok=True)

    def worker(k):
        client = OpenAI(base_url=urls[k % len(urls)], api_key='EMPTY', timeout=3600, max_retries=0)
        while True:
            try:
                bench, it = q.get_nowait()
            except queue.Empty:
                return
            try:
                ans = request_answer(client, it, data_dir)
                with locks[bench]:
                    with answers_path(run_dir, bench).open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'question_id': it['question_id'], 'model_answer': ans}, ensure_ascii=False) + '\n')
                with stat_lock:
                    stats['done'] += 1
                    stats['errors'] += not valid_answer(ans)
            except Exception as e:
                with stat_lock:
                    stats['errors'] += 1
                print(f'[infer] {bench} {it.get("question_id")}: {type(e).__name__}: {str(e)[:200]}', file=sys.stderr, flush=True)

    ts = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(threads * len(urls))]
    for t in ts:
        t.start()
    t0, last = time.time(), time.time()
    while any(t.is_alive() for t in ts):
        time.sleep(5)
        if time.time() - last >= 60:
            el = time.time() - t0
            rate = stats['done'] / max(el, 1)
            left = len(work) - stats['done']
            print(f'[infer] {stats["done"]}/{len(work)} answered, {rate:.1f}/s, about {left / max(rate, 1e-6) / 60:.0f} min left, errors {stats["errors"]}', flush=True)
            last = time.time()
    print(f'[infer] round finished: {stats["done"]} answered in {(time.time() - t0) / 60:.1f} min, errors {stats["errors"]}', flush=True)


def cmd_infer(a):
    benches = parse_benchmarks(a.benchmarks)
    work = pending(a.data_dir, a.run_dir, benches)
    if a.count_pending:
        print(len(work))
        return
    if not work:
        print('[infer] all answers present', flush=True)
        return
    urls = [u for u in a.api_bases.split(',') if u]
    if not urls:
        raise SystemExit('--api-bases is required when answers are missing')
    raise_nofile()
    for rnd in range(3):
        print(f'[infer] {len(work)} questions to answer on {len(urls)} server(s)', flush=True)
        run_round(work, urls, a.threads, a.data_dir, a.run_dir)
        work = pending(a.data_dir, a.run_dir, benches)
        if not work:
            break
    if work:
        print(f'[infer] WARNING: {len(work)} questions still unanswered; rerun the same command to resume', file=sys.stderr, flush=True)


def main_score(metrics, bench):
    return metrics.get(MAIN_METRIC.get(bench, 'score'))


def mean(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    return sum(v) / len(v) if v else None


def summarize(name, model, results, benches):
    table = {b: main_score(results[b], b) if b in results else None for b in BENCHMARKS}
    perception = mean([table[b] for b in PERCEPTION])
    reasoning = mean([table[b] for b in REASONING])
    overall = mean([table[b] for b in BENCHMARKS])
    missing = {b: results[b]['n_expected'] - results[b]['n_answered'] for b in benches if b in results and results[b]['n_answered'] < results[b]['n_expected']}
    missing.update({b: 'not scored' for b in benches if b not in results})
    return {'name': name, 'model': model, 'scores': {TITLES[b]: table[b] for b in BENCHMARKS},
            'perception_mean': perception, 'reasoning_mean': reasoning, 'overall_mean': overall,
            'complete': not missing and all(table[b] is not None for b in BENCHMARKS), 'missing': missing,
            'judge_failed': sum(results[b]['judge']['judge_failed'] for b in results),
            'judge_rate': {TITLES[b]: results[b]['judge']['judge_rate'] for b in JUDGED if b in results},
            'details': results}


def fmt(v):
    return '-' if v is None else f'{v:.2f}'


def table_lines(summaries):
    head = [TITLES[b] for b in BENCHMARKS] + ['Perception', 'Reasoning', 'Overall']
    lines = ['| Model | ' + ' | '.join(head) + ' |', '|---|' + '---|' * len(head)]
    for s in summaries:
        vals = [s['scores'][TITLES[b]] for b in BENCHMARKS] + [s['perception_mean'], s['reasoning_mean'], s['overall_mean']]
        lines.append(f'| {s["name"]}{"" if s["complete"] else " (incomplete)"} | ' + ' | '.join(fmt(v) for v in vals) + ' |')
    return lines


def cmd_score(a):
    benches = parse_benchmarks(a.benchmarks)
    run_dir = Path(a.run_dir)
    judge = scoring.Judge(None if a.count_pending else a.judge_base, run_dir / 'judge_cache.jsonl', model=a.judge_model)
    results = {}
    for bench in [b for b in BENCHMARKS if b in benches]:
        if a.count_pending and bench not in JUDGED:
            continue
        items = load_items(a.data_dir, bench)
        ans = read_answers(run_dir, bench)
        rows = [dict(it, model_answer=ans[it['question_id']]) for it in items if it['question_id'] in ans]
        if not rows:
            continue
        per, metrics = scoring.SCORERS[bench](rows, judge)
        if a.count_pending:
            continue
        metrics['judge'] = scoring.judge_stats(per)
        metrics['n_answered'] = len(rows)
        metrics['n_expected'] = len(items)
        results[bench] = metrics
        p = run_dir / 'per_sample' / f'{bench}.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(x, ensure_ascii=False, default=str) + '\n' for x in per), encoding='utf-8')
        j = metrics['judge']
        print(f'[score] {TITLES[bench]}: {main_score(metrics, bench):.2f} (rule {j["rule"]}, judge {j["judge"]}, judge failed {j["judge_failed"]})', flush=True)
    if a.count_pending:
        print(len(judge.misses))
        return
    s = summarize(a.name or run_dir.name, a.model, results, benches)
    (run_dir / 'summary.json').write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    lines = table_lines([s])
    note = []
    if s['judge_failed']:
        note.append(f'{s["judge_failed"]} answers needed the judge but got no judge output and were scored as wrong; rerun with the judge to fix.')
    if s['missing']:
        note.append(f'incomplete: {s["missing"]}')
    (run_dir / 'report.md').write_text('\n'.join(lines + [''] + note) + '\n', encoding='utf-8')
    print('\n'.join([''] + lines + note), flush=True)


def cmd_report(a):
    summaries = [json.loads((Path(d) / 'summary.json').read_text(encoding='utf-8')) for d in a.run_dirs]
    print('\n'.join(table_lines(summaries)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--benchmarks', default=','.join(BENCHMARKS))
    p.add_argument('--api-bases', default='')
    p.add_argument('--threads', type=int, default=288)
    p.add_argument('--count-pending', action='store_true')
    p = sub.add_parser('score')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--benchmarks', default=','.join(BENCHMARKS))
    p.add_argument('--judge-base', default='')
    p.add_argument('--judge-model', default='judge')
    p.add_argument('--name', default='')
    p.add_argument('--model', default='')
    p.add_argument('--count-pending', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('run_dirs', nargs='+')
    a = ap.parse_args()
    {'infer': cmd_infer, 'score': cmd_score, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
