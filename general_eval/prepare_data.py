import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

BENCHMARKS = ['mmstar', 'cvbench', 'mmvp', 'pope', 'hallusionbench', 'mathvista', 'mathverse', 'mathvision', 'wemath', 'dynamath']

SOURCES = {
    'mmstar': ('Lin-Chen/MMStar', 'bc98d668301da7b14f648724866e57302778ab27', ['mmstar.parquet']),
    'cvbench': ('nyu-visionx/CV-Bench', 'bc284db50d036958861cb60cdd7b77612052ce0d', ['test_2d.parquet', 'test_3d.parquet']),
    'mmvp': ('MMVP/MMVP', '37eafecab8a3940c50c2ade5b36de69dbc99a8cf', ['Questions.csv', 'MMVP Images/*.jpg']),
    'pope': ('lmms-lab-encoder/POPE', '4db1276663dfa5eb8ad16a52d24c31a09e470896',
             ['Full/adversarial-00000-of-00001.parquet', 'Full/popular-00000-of-00001.parquet', 'Full/random-00000-of-00001.parquet']),
    'hallusionbench': ('lmms-lab-encoder/HallusionBench', 'cd417161857aefb23d878d42cf1bb53aa9dd646f',
                       ['data/image-00000-of-00001.parquet', 'data/non_image-00000-of-00001.parquet']),
    'mathvista': ('AI4Math/MathVista', '2b6ad69445fbb5695c9b165475e8decdbeb97747', ['data/testmini-00000-of-00001-725687bf7a18d64b.parquet']),
    'mathverse': ('AI4Math/MathVerse', '3bc86196678bad115a923d2851c6821dbe235939', ['testmini.parquet']),
    'mathvision': ('MathLLMs/MathVision', '2837ddb3f13abaf6b3997c12d80753e5470bd46a', ['data/test-00000-of-00001-3532b8d3f1b4047a.parquet']),
    'wemath': ('We-Math/We-Math', '527c44d4d94c4e3c7c98157460146a8b18c8420a', ['data/testmini-00000-of-00001-adbfe22dcd3558e6.parquet']),
    'dynamath': ('DynaMath/DynaMath_Sample', 'bc7fcd1ae58caa445d61f6dc0c241b0bb8a7f4c6',
                 ['data/sample1-00000-of-00001.parquet', 'data/sample10-00000-of-00010.parquet', 'data/sample2-00000-of-00002.parquet',
                  'data/sample3-00000-of-00003.parquet', 'data/sample4-00000-of-00004.parquet', 'data/sample5-00000-of-00005.parquet',
                  'data/sample6-00000-of-00006.parquet', 'data/sample7-00000-of-00007.parquet', 'data/sample8-00000-of-00008.parquet',
                  'data/sample9-00000-of-00009.parquet']),
}

COUNTS = {'mmstar': 1500, 'cvbench': 2638, 'mmvp': 300, 'pope': 9000, 'hallusionbench': 1129,
          'mathvista': 1000, 'mathverse': 3940, 'mathvision': 3040, 'wemath': 1740, 'dynamath': 5010}

FINGERPRINTS = {}

POPE_SUFFIX = '\nAnswer the question using a single word or phrase.'
CVBENCH_EXT = "\nAnswer with the option's letter from the given choices directly."
MMSTAR_TAIL = '\nPlease select the correct answer from the options above. \n'
MATHV_HEAD = 'Please solve the problem and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'
WEMATH_TEMPLATE = ('Now, we require you to solve a multiple-choice math question. Please briefly describe your thought process and provide the final answer(option).\n'
                   'Question: {question}\n'
                   'Option: {option}\n'
                   'Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n'
                   '<Thought process>: <<your thought process>> <Answer>: <<your option>>')
DYNAMATH_GUIDE = (
    "\n## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere \n"
    "to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain \n"
    "detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. If the problem is a "
    "multiple choice problem, just provide the corresponing choice option, \n"
    "such as 'A', 'B', 'C', or 'D'. If the answer is a numerical value, format it as a three-digit floating-point number.\n\n"
    "Example of expected JSON response format:\n\n")
DYNAMATH_EXAMPLE = json.dumps({'solution': '[Detailed step-by-step explanation]', 'short answer': '[Concise Answer]'}, indent=4)

EXT = {'JPEG': '.jpg', 'MPO': '.jpg', 'PNG': '.png', 'GIF': '.gif', 'BMP': '.bmp', 'WEBP': '.webp', 'TIFF': '.tif'}


def mathv_options(opts):
    if not opts or ''.join(opts) == 'ABCDE':
        return ''
    return ''.join(f'({chr(65 + i)}) {o}\n' for i, o in enumerate(opts))


def build_query(bench, x):
    if bench == 'mmstar':
        return f'Question: {x["question"]}' + MMSTAR_TAIL
    if bench == 'cvbench':
        return x['prompt'] + CVBENCH_EXT
    if bench == 'mmvp':
        return f'{x["question"]} {x["options"]}'
    if bench in ('pope', 'hallusionbench', 'mathvista'):
        return x['query']
    if bench == 'mathverse':
        return x['query_cot']
    if bench == 'mathvision':
        return MATHV_HEAD + f'{x["question"]}\n{mathv_options(x.get("options"))}'
    if bench == 'wemath':
        return WEMATH_TEMPLATE.format(question=x['question'], option=x['option'])
    if bench == 'dynamath':
        q = '## Question\n' + x['question']
        if x.get('answer_type') == 'float':
            q += 'Please answer in a floating-point number.'
        return q + DYNAMATH_GUIDE + DYNAMATH_EXAMPLE
    raise KeyError(bench)


class ImageStore:
    def __init__(self, data_dir, bench):
        self.data_dir = Path(data_dir)
        self.rel = Path('images') / bench
        (self.data_dir / self.rel).mkdir(parents=True, exist_ok=True)
        self.sha = {}

    def put(self, stem, b):
        from PIL import Image
        with Image.open(io.BytesIO(b)) as im:
            fmt = im.format
            im.verify()
        if fmt not in EXT:
            raise SystemExit(f'unknown image format {fmt} for {stem}')
        name = re.sub(r'[^A-Za-z0-9._-]', '_', stem) + EXT[fmt]
        h = hashlib.sha256(b).hexdigest()
        if name in self.sha:
            if self.sha[name] != h:
                raise SystemExit(f'two different images share the name {name}')
            return str(self.rel / name)
        p = self.data_dir / self.rel / name
        if not (p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h):
            tmp = p.with_name(p.name + '.tmp')
            tmp.write_bytes(b)
            os.replace(tmp, p)
        self.sha[name] = h
        return str(self.rel / name)


def parquet_rows(path):
    import pyarrow.parquet as pq
    for rb in pq.ParquetFile(path).iter_batches(batch_size=64):
        yield from rb.to_pylist()


def img_bytes(v, where):
    if isinstance(v, (bytes, bytearray)) and v:
        return bytes(v)
    if isinstance(v, dict) and v.get('bytes'):
        return v['bytes']
    raise SystemExit(f'{where} has no embedded image bytes')


def clean(r, drop):
    out = {}
    for k, v in r.items():
        if k in drop or isinstance(v, (bytes, bytearray)):
            continue
        if isinstance(v, dict) and 'bytes' in v:
            continue
        out[k] = v
    return out


def conv_mmstar(raw, store):
    out = []
    for r in parquet_rows(raw / 'mmstar.parquet'):
        x = clean(r, {'image'})
        x.update({'question_id': f'mmstar-{r["index"]}', 'images': [store.put(str(r['index']), img_bytes(r['image'], f'MMStar {r["index"]}'))],
                  'response': str(r['answer']).strip(), 'category': r['category']})
        out.append(x)
    return out


def conv_cvbench(raw, store):
    out = []
    for path in ('test_2d.parquet', 'test_3d.parquet'):
        for r in parquet_rows(raw / path):
            stem = r['filename'].replace('img/', '').rsplit('.', 1)[0].replace('/', '__')
            x = clean(r, {'image'})
            ans = r['answer'].strip()
            x.update({'question_id': f'cvbench-{r["idx"]:04d}', 'images': [store.put(stem, img_bytes(r['image'], f'CV-Bench {stem}'))],
                      'response': ans.strip('()'), 'answer_raw': ans, 'category': f'{r["type"]}-{r["source"]}-{r["task"]}'})
            out.append(x)
    return out


def conv_mmvp(raw, store):
    out = []
    with open(raw / 'Questions.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            i = int(r['Index'])
            img = (raw / 'MMVP Images' / f'{i}.jpg').read_bytes()
            ans = r['Correct Answer'].strip()
            out.append({'question_id': f'mmvp-{i}', 'images': [store.put(str(i), img)], 'question': r['Question'].strip(),
                        'options': r['Options'].strip(), 'response': ans.strip('()').lower(), 'answer_raw': ans,
                        'pair_id': (i - 1) // 2, 'index': i, 'category': 'mmvp'})
    return out


def conv_pope(raw, store):
    out = []
    for s in ('adversarial', 'popular', 'random'):
        for r in parquet_rows(raw / 'Full' / f'{s}-00000-of-00001.parquet'):
            if r['category'] != s:
                raise SystemExit(f'POPE {s}: unexpected category {r["category"]!r}')
            ans = str(r['answer']).strip().lower()
            if ans not in ('yes', 'no'):
                raise SystemExit(f'POPE {s} question_id={r["question_id"]}: answer {r["answer"]!r}')
            b = (r.get('image') or {}).get('bytes')
            if not b:
                raise SystemExit(f'POPE {s} question_id={r["question_id"]} has no image')
            out.append({'question_id': f'pope-{s}-{r["question_id"]}', 'images': [store.put(r['image_source'], b)], 'query': r['question'].strip() + POPE_SUFFIX,
                        'response': ans, 'category': s, 'image_source': r['image_source'], 'orig_id': r.get('id'), 'orig_question_id': r['question_id']})
    return out


def hb_row(r, images):
    a = str(r['gt_answer']).strip()
    if a not in ('0', '1'):
        raise SystemExit(f'HallusionBench {r["category"]}/{r["subcategory"]} set {r["set_id"]} q {r["question_id"]}: gt_answer {a!r}')
    row = {'question_id': f'hb-{r["category"]}-{r["subcategory"]}-{r["set_id"]}-{r["figure_id"]}-{r["question_id"]}', 'images': images,
           'query': r['question'].strip(), 'response': 'yes' if a == '1' else 'no'}
    for k in ('category', 'subcategory', 'visual_input', 'set_id', 'figure_id', 'question_id', 'sample_note', 'gt_answer', 'gt_answer_details', 'filename'):
        row['orig_question_id' if k == 'question_id' else k] = r.get(k)
    return row


def conv_hallusionbench(raw, store):
    out = []
    for r in parquet_rows(raw / 'data' / 'image-00000-of-00001.parquet'):
        if not r.get('filename'):
            raise SystemExit(f'HallusionBench image row without filename (set {r.get("set_id")} q {r.get("question_id")})')
        stem = re.sub(r'\.[A-Za-z]+$', '', r['filename'].lstrip('./')).replace('/', '__')
        b = (r.get('image') or {}).get('bytes')
        if not b:
            raise SystemExit(f'HallusionBench {r["filename"]} has no image')
        out.append(hb_row(r, [store.put(stem, b)]))
    for r in parquet_rows(raw / 'data' / 'non_image-00000-of-00001.parquet'):
        if r.get('image'):
            raise SystemExit('HallusionBench non_image row with an image')
        out.append(hb_row(r, []))
    return out


def conv_mathvista(raw, store):
    out = []
    for path in SOURCES['mathvista'][2]:
        for r in parquet_rows(raw / path):
            x = clean(r, {'decoded_image'})
            x.update({'question_id': f'mathvista-{r["pid"]}', 'images': [store.put(str(r['pid']), img_bytes(r['decoded_image'], f'MathVista {r["pid"]}'))],
                      'response': str(r['answer']).strip(), 'category': (r.get('metadata') or {}).get('task') or r.get('task')})
            out.append(x)
    return out


def conv_mathverse(raw, store):
    out = []
    for r in parquet_rows(raw / 'testmini.parquet'):
        if not (isinstance(r.get('image'), dict) and r['image'].get('bytes')):
            raise SystemExit(f'MathVerse {r["sample_index"]} has no image')
        x = clean(r, {'image'})
        x.update({'question_id': f'mathverse_testmini-{r["sample_index"]}', 'response': str(r['answer']).strip(), 'category': r['problem_version'],
                  'images': [store.put(f's{r["sample_index"]}', r['image']['bytes'])]})
        out.append(x)
    return out


def conv_mathvision(raw, store):
    out = []
    for path in SOURCES['mathvision'][2]:
        for r in parquet_rows(raw / path):
            x = clean(r, {'decoded_image'})
            x.update({'question_id': f'mathvision_test-{r["id"]}', 'images': [store.put(str(r['id']), img_bytes(r['decoded_image'], f'MathVision {r["id"]}'))],
                      'response': str(r['answer']).strip(), 'category': r.get('subject')})
            out.append(x)
    return out


def conv_wemath(raw, store):
    out = []
    for path in SOURCES['wemath'][2]:
        for k, r in enumerate(parquet_rows(raw / path)):
            v = r['image_path']
            stem = f'{str(r["key"]).split("_")[0]}__{Path((v or {}).get("path") or f"row{k:04d}").stem}'
            x = clean(r, {'image_path'})
            x.update({'question_id': f'wemath-{k:04d}', 'images': [store.put(stem, img_bytes(v, f'We-Math row {k}'))],
                      'response': str(r['answer']).strip(), 'category': r.get('key'), 'image_name': (v or {}).get('path')})
            out.append(x)
    return out


def conv_dynamath(raw, store):
    out = []
    for path in SOURCES['dynamath'][2]:
        k = int(Path(path).name.split('-')[0].replace('sample', ''))
        for r in parquet_rows(raw / path):
            x = clean(r, {'decoded_image'})
            x.update({'question_id': f'dynamath-v{k}-{r["id"]}', 'images': [store.put(f'v{k}_{r["id"]}', img_bytes(r['decoded_image'], f'DynaMath v{k} {r["id"]}'))],
                      'response': str(r['ground_truth']).strip(), 'category': r.get('subject'), 'variant': k, 'seed_id': r['id']})
            out.append(x)
    return out


CONVERT = {'mmstar': conv_mmstar, 'cvbench': conv_cvbench, 'mmvp': conv_mmvp, 'pope': conv_pope, 'hallusionbench': conv_hallusionbench,
           'mathvista': conv_mathvista, 'mathverse': conv_mathverse, 'mathvision': conv_mathvision, 'wemath': conv_wemath, 'dynamath': conv_dynamath}


def fingerprint(data_dir, rows):
    h = hashlib.sha256()
    for r in rows:
        x = dict(r, images=[hashlib.sha256((Path(data_dir) / p).read_bytes()).hexdigest() for p in r['images']])
        h.update(json.dumps(x, ensure_ascii=False, sort_keys=True).encode() + b'\n')
    return h.hexdigest()


def download(bench, raw):
    from huggingface_hub import snapshot_download
    repo, revision, files = SOURCES[bench]
    print(f'[data] {bench}: downloading {repo}@{revision[:8]}', flush=True)
    snapshot_download(repo_id=repo, repo_type='dataset', revision=revision, allow_patterns=files, local_dir=str(raw))


def missing_raw(bench, raw):
    if bench == 'mmvp':
        return [p for p in ('Questions.csv', 'MMVP Images/1.jpg') if not (raw / p).exists()]
    return [p for p in SOURCES[bench][2] if not (raw / p).exists()]


def prepare(bench, data_dir, raw_root, allow_download, force):
    out_path = data_dir / f'{bench}.json'
    if out_path.exists() and not force:
        print(f'[data] {bench}: {out_path} exists', flush=True)
        return True
    raw = raw_root / bench
    if missing_raw(bench, raw):
        if not allow_download:
            raise SystemExit(f'{bench}: missing {missing_raw(bench, raw)} under {raw}')
        download(bench, raw)
        if missing_raw(bench, raw):
            raise SystemExit(f'{bench}: download incomplete, still missing {missing_raw(bench, raw)}')
    rows = CONVERT[bench](raw, ImageStore(data_dir, bench))
    for x in rows:
        x['query'] = build_query(bench, x)
    ids = [r['question_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f'{bench}: duplicate question ids')
    if len(rows) != COUNTS[bench]:
        raise SystemExit(f'{bench}: {len(rows)} records, expected {COUNTS[bench]}')
    fp = fingerprint(data_dir, rows)
    tmp = out_path.with_name(out_path.name + '.tmp')
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, out_path)
    want = FINGERPRINTS.get(bench)
    same = want is None or want == fp
    print(f'[data] {bench}: {len(rows)} records -> {out_path}; fingerprint {fp[:16]} '
          + ('matches the paper data' if want == fp else 'not checked' if want is None else f'DIFFERS from the paper data ({want[:16]})'), flush=True)
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(Path(__file__).resolve().parent.parent / 'data' / 'general'))
    ap.add_argument('--raw-dir', default=None)
    ap.add_argument('--benchmarks', default=','.join(BENCHMARKS))
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--print-fingerprints', action='store_true')
    a = ap.parse_args()
    data_dir = Path(a.data_dir).resolve()
    raw_root = Path(a.raw_dir).resolve() if a.raw_dir else data_dir / 'raw'
    benches = [b for b in a.benchmarks.split(',') if b]
    unknown = [b for b in benches if b not in SOURCES]
    if unknown:
        raise SystemExit(f'unknown benchmarks: {unknown}')
    data_dir.mkdir(parents=True, exist_ok=True)
    if a.print_fingerprints:
        for b in benches:
            rows = json.loads((data_dir / f'{b}.json').read_text(encoding='utf-8'))
            print(json.dumps({b: fingerprint(data_dir, rows)}))
        return
    bad = [b for b in benches if not prepare(b, data_dir, raw_root, not a.no_download, a.force)]
    if bad:
        print(f'[data] WARNING: {",".join(bad)} differ from the data used in the paper; scores may not match', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
