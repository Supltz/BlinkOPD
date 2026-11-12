import concurrent.futures as cf
import hashlib
import json
import math
import re
import signal
import threading
from collections import Counter, defaultdict
from pathlib import Path


def strip_think(text):
    text = text or ''
    i = text.rfind('</think>')
    return text[i + len('</think>'):].strip() if i != -1 else text.strip()


class Judge:
    def __init__(self, url, cache_path, model='judge', max_tokens=2048, workers=64):
        self.url, self.model, self.max_tokens, self.workers = url, model, max_tokens, workers
        self.cache_path = Path(cache_path)
        self.lock = threading.Lock()
        self.cache = {}
        self.misses = set()
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r['key']] = r['output']
        self.client = None
        if url:
            from openai import OpenAI
            self.client = OpenAI(base_url=url, api_key='EMPTY', timeout=600)

    @staticmethod
    def key(task, messages):
        return hashlib.sha256(json.dumps([task, messages], ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def ask(self, task, messages):
        key = self.key(task, messages)
        if key in self.cache:
            return self.cache[key]
        if self.client is None:
            with self.lock:
                self.misses.add(key)
            return None
        out = None
        for _ in range(3):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0, max_tokens=self.max_tokens)
                out = (r.choices[0].message.content or '').strip()
                break
            except Exception:
                continue
        if out is None:
            with self.lock:
                self.misses.add(key)
            return None
        with self.lock:
            self.cache[key] = out
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'key': key, 'task': task, 'output': out}, ensure_ascii=False) + '\n')
        return out

    def map(self, task, items, fn):
        if not items:
            return []
        with cf.ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda it: self.ask(task, fn(it)), items))


def user(text):
    return [{'role': 'user', 'content': text}]


def marker_answer(text):
    t = strip_think(text)
    if not t:
        return None
    i = t.rfind('\\boxed{')
    if i != -1:
        j, depth, k = i + len('\\boxed{'), 1, i + len('\\boxed{')
        while k < len(t) and depth:
            depth += {'{': 1, '}': -1}.get(t[k], 0)
            k += 1
        if depth == 0:
            return t[j:k - 1].strip()
    best = None
    for m in re.finditer(r'(?i)(?:the\s+)?(?:final\s+|correct\s+)?answer\s+is\s*:?|(?<![a-z])answer\s*:|答案(?:是|为)?[:：]', t):
        best = m
    if best is not None:
        tail = t[best.end():].strip().split('\n')[0].strip().strip('*').strip()
        tail = re.sub(r'^[\s:：*]+', '', tail).rstrip('.。').strip()
        if tail:
            return tail
    last = t.strip().split('\n')[-1].strip().strip('*').strip()
    if 0 < len(last) <= 20:
        return last.rstrip('.。')
    return None


def md_clean(text):
    return re.sub(r'[*`#✅]+', '', text or '')


def lenient_letter(text, letters='ABCDEF'):
    t = md_clean(strip_think(text)).strip()
    letter = option_letter(marker_answer(t), letters)
    if letter is None:
        m = re.match(rf'^\(?([{letters}])\)?(?:[.:：)]|$)', t)
        letter = m.group(1) if m else None
    return letter


def first_number(s):
    if s is None:
        return None
    m = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', s.replace('−', '-'))
    return m.group(0).replace(',', '') if m else None


def option_letter(s, letters='ABCDEFGH'):
    if s is None:
        return None
    s = s.strip()
    m = re.match(rf'^\(?([{letters}])\)?(?:[\s.:：,)]|$)', s)
    return m.group(1) if m else None


def cambrian_cvbench_extract(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    text = text.strip().rstrip('.').lstrip('(').rstrip(')')
    if len(text) > 1:
        text = text[0]
    return text


def mcq_judge(judge, task, ask, question_of, gt_of):
    from official import vlmeval_mcq as vm
    outs = judge.map(task, ask, lambda a: user(vm.build_prompt(question_of(a[1]), vm.build_option_str(a[2]), strip_think(a[1]['model_answer'])))) \
        if (judge and ask) else [None] * len(ask)
    for (x, r, ch), o in zip(ask, outs):
        if o is None:
            x.update(extracted=None, correct=False, source='judge_failed')
        else:
            opt = vm.can_infer(o, ch) or 'Z'
            x.update(extracted=opt, correct=opt == gt_of(r), source='judge')


def score_cvbench(rows, judge=None):
    per, ask = [], []
    for r in rows:
        off = cambrian_cvbench_extract(strip_think(r['model_answer']))
        gt = r['answer'].strip()[1].upper()
        ch = {chr(65 + i): str(c) for i, c in enumerate(r['choices'])}
        x = {'question_id': r['question_id'], 'source_ds': r['source'], 'task': r['task'], 'extracted_official': off, 'correct_official': off == gt.lower()}
        letter = lenient_letter(r['model_answer'], ''.join(ch))
        if letter:
            x.update(extracted=letter, correct=letter == gt, source='rule')
        else:
            ask.append((x, r, ch))
        per.append(x)
    mcq_judge(judge, 'cvbench', ask, lambda r: r['question'], lambda r: r['answer'].strip()[1].upper())

    def acc(f, k='correct'):
        return _mean([p[k] for p in per if f(p)])

    def overall(k):
        ade, coco, omni = (acc(lambda p, d=d: p['source_ds'] == d, k) for d in ('ADE20K', 'COCO', 'Omni3D'))
        return 100 * 0.5 * ((ade + coco) / 2 + omni), (ade, coco, omni)
    main, (ade, coco, omni) = overall('correct')
    official, _ = overall('correct_official')
    return per, {'score': main, 'ADE20K': 100 * ade, 'COCO': 100 * coco, 'Omni3D': 100 * omni,
                 'score_official': official,
                 'pooled': 100 * _mean([p['correct'] for p in per]),
                 'by_task': {t: 100 * acc(lambda p, t=t: p['task'] == t) for t in sorted({p['task'] for p in per})}}


def mmstar_correct(predict, answers):
    answer = answers.lower().strip().replace('\n', ' ')
    predict = predict.lower().strip().replace('\n', ' ')
    try:
        if answer == predict[0]:
            return True
        elif predict[0] == '(' and answer == predict[1]:
            return True
        elif predict[0:7] == 'option ' and answer == predict[7]:
            return True
        elif predict[0:14] == 'the answer is ' and answer == predict[14]:
            return True
    except Exception:
        pass
    return False


def mmstar_choices(question):
    i = question.rfind('Options:')
    if i < 0:
        return {}
    parts = re.split(r'(?:^|,)\s*([A-D]):\s', question[i + len('Options:'):])
    return {parts[k]: parts[k + 1].strip() for k in range(1, len(parts) - 1, 2)}


def score_mmstar(rows, judge=None):
    per, ask = [], []
    for r in rows:
        gt = str(r['answer']).strip().upper()
        ch = mmstar_choices(r['question']) or {c: '' for c in 'ABCD'}
        x = {'question_id': r['question_id'], 'category': r['category'], 'l2': r['l2_category'],
             'correct_official': mmstar_correct(str(strip_think(r['model_answer'])), str(r['answer']))}
        letter = lenient_letter(r['model_answer'], ''.join(ch))
        if letter:
            x.update(extracted=letter, correct=letter == gt, source='rule')
        else:
            ask.append((x, r, ch))
        per.append(x)
    mcq_judge(judge, 'mmstar', ask, lambda r: r['question'].split('\nOptions:')[0], lambda r: str(r['answer']).strip().upper())
    cats = sorted({p['category'] for p in per})
    return per, {'score': 100 * sum(p['correct'] for p in per) / 1500.0,
                 'score_official': 100 * sum(p['correct_official'] for p in per) / 1500.0,
                 'by_category': {c: 100 * _mean([p['correct'] for p in per if p['category'] == c]) for c in cats}}


class _Timeout(Exception):
    pass


def _with_timeout(fn, sec, *a):
    if threading.current_thread() is not threading.main_thread():
        return fn(*a)

    def h(*_):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, h)
    signal.alarm(sec)
    try:
        return fn(*a)
    except _Timeout:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def mathv_model_answer(response):
    from official.mathv_utils import find_math_answer, is_number
    model_answer = response.strip()
    for c in 'ABCDE':
        if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
            model_answer = c
    if is_number(model_answer.split('is ')[-1].rstrip('.')):
        model_answer = model_answer.split('is ')[-1].rstrip('.')
    if 'oxed{' not in model_answer:
        for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                model_answer = model_answer.split('\n')[0].split('. ')[0]
            flag = flag.replace('the', 'The')
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                model_answer = model_answer.split('\n')[0].split('. ')[0]
    elif model_answer.count('oxed{') > 1:
        model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
    model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
    return model_answer


def score_mathvision(rows, judge=None):
    from official.mathv_utils import is_equal
    per = []
    for r in rows:
        gt_answer = str(r['answer'])
        opts = r.get('options') or []
        gt_answer_value = opts[ord(gt_answer) - ord('A')] if len(opts) > 0 and len(gt_answer) == 1 and 0 <= ord(gt_answer) - ord('A') < len(opts) else ''
        try:
            ma = mathv_model_answer(strip_think(r['model_answer']))
        except Exception:
            ma = ''
        ok = bool(_with_timeout(is_equal, 5, gt_answer, ma) or _with_timeout(is_equal, 5, gt_answer_value, ma))
        per.append({'question_id': r['question_id'], 'id': str(r['id']), 'extracted': ma, 'correct': ok, 'source': 'rule', 'subject': r.get('subject'), 'level': r.get('level')})
    return per, {'score': 100 * _mean([p['correct'] for p in per]), 'n': len(per)}


def score_wemath(rows, judge=None):
    import pandas as pd
    df = pd.DataFrame([{'ID': r['ID'], 'key': r['key'], 'answer': r['answer'], 'knowledge concept': r['knowledge concept'],
                        'response': strip_think(r['model_answer']), 'question_id': r['question_id']} for r in rows])
    df['processed_answer'] = df['response'].str.split('Answer').str[-1].str.strip().str.replace(r'[>><<:.]', '', regex=True).str.strip()
    df['processed_answer'] = df['processed_answer'].apply(lambda x: x[0] if x and x[0] in 'ABCDEFGH' else None)
    df['joker'] = df['processed_answer'] == df['answer']

    def process_steps_data(df, steps):
        steps_data = {f'{steps}steps_{i}': df[df['key'] == f'{steps}steps_{i}'] for i in range(1, steps + 1)}
        steps_data[f'{steps}steps_multi'] = df[df['key'] == f'{steps}steps_multi']
        for key, data in steps_data.items():
            data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
        merged_data = steps_data[f'{steps}steps_1']
        for i in range(2, steps + 1):
            merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_{i}'], left_on=f'ID_1', right_on=f'ID_{i}', how='left')
        merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_multi'], left_on=f'ID_1', right_on='ID_multi', how='left')
        return merged_data

    m2 = process_steps_data(df[df['key'].str.contains('2steps')].copy(), 2)
    m3 = process_steps_data(df[df['key'].str.contains('3steps')].copy(), 3)

    def j(m, k):
        return m[f'joker_{k}'] == True

    def nj(m, k):
        return m[f'joker_{k}'] == False
    rm_loose = len(m2[(nj(m2, 1) & nj(m2, 2)) & j(m2, 'multi')]) + len(m3[(nj(m3, 1) & nj(m3, 2) & nj(m3, 3)) & j(m3, 'multi')])
    rm_strict = len(m2[(nj(m2, 1) | nj(m2, 2)) & j(m2, 'multi')]) + len(m3[(nj(m3, 1) | nj(m3, 2) | nj(m3, 3)) & j(m3, 'multi')])
    ig = len(m2[(j(m2, 1) & j(m2, 2)) & nj(m2, 'multi')]) + len(m3[(j(m3, 1) & j(m3, 2) & j(m3, 3)) & nj(m3, 'multi')])
    ik = len(m2[(nj(m2, 1) | nj(m2, 2)) & nj(m2, 'multi')]) + len(m3[(nj(m3, 1) | nj(m3, 2) | nj(m3, 3)) & nj(m3, 'multi')])
    cm_loose = len(m2[(j(m2, 1) | j(m2, 2)) & j(m2, 'multi')]) + len(m3[(j(m3, 1) | j(m3, 2) | j(m3, 3)) & j(m3, 'multi')])
    cm_strict = len(m2[(j(m2, 1) & j(m2, 2)) & j(m2, 'multi')]) + len(m3[(j(m3, 1) & j(m3, 2) & j(m3, 3)) & j(m3, 'multi')])
    total = len(m2) + len(m3)
    per = [{'question_id': q, 'extracted': a, 'correct': bool(c), 'source': 'rule'} for q, a, c in zip(df['question_id'], df['processed_answer'], df['joker'])]
    return per, {'score': 100 * (525 - 0.5 * ig - rm_strict - ik) / 525, 'score_loose': 100 * (525 - 0.5 * ig - rm_loose - ik) / 525,
                 'IK': ik, 'IG': ig, 'CM_strict': cm_strict, 'RM_strict': rm_strict, 'CM_loose': cm_loose, 'RM_loose': rm_loose,
                 'n_composite': total, 'question_acc': 100 * _mean([p['correct'] for p in per])}


def dynamath_preprocess(str1):
    str0 = str1
    if str0 == "":
        return str1
    while str0[0] != "{":
        str0 = str0[1:]
        if len(str0) == 1:
            return str1
    while str0[-1] != "}":
        str0 = str0[:-1]
        if len(str0) == 1:
            return str1
    str2 = str0.replace("\\", "")
    str2 = str2.replace("\\n", "\n")
    return str2


def dynamath_transfer(str1):
    if "π" in str1:
        strs = str1.split('π')
        str1 = strs[0]
        return float(str1) * math.pi
    else:
        return float(str1)


def dynamath_correct(content, gt, answer_type):
    try:
        description = dynamath_preprocess(content)
    except Exception:
        description = content
    try:
        dj = json.loads(description, strict=False)
    except Exception:
        dj = {"solution": description, "short answer": description.split('\n')[-1][:-1]}
    try:
        answer = str(dj.get("short answer"))
        if answer_type == "float":
            if not answer.isdigit():
                parts = answer.split(' ')
                answer = parts[0]
                answer = dynamath_transfer(answer)
            diff = float(answer) - float(gt)
            return abs(diff) <= 0.001, str(answer)
        elif answer_type == "multiple choice":
            if len(answer) == 1:
                return answer == gt, answer
            else:
                return gt in answer[0:3], answer
        else:
            return gt in answer, answer
    except Exception:
        return False, None


def score_dynamath(rows, judge=None):
    per = []
    for r in rows:
        ok, a = dynamath_correct(strip_think(r['model_answer']), str(r['ground_truth']), r['answer_type'])
        per.append({'question_id': r['question_id'], 'extracted': a, 'correct': bool(ok), 'source': 'rule', 'seed': r['seed_id'], 'variant': r['variant']})
    by_seed = defaultdict(list)
    for p in per:
        by_seed[p['seed']].append(p['correct'])
    return per, {'score': 100 * _mean([p['correct'] for p in per]), 'worst_case': 100 * _mean([all(v) for v in by_seed.values()]), 'n_seeds': len(by_seed)}


def pope_yes(text):
    if text.find('.') != -1:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        return False
    return True


def score_pope(rows, judge=None):
    per, by = [], defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        pred, lab = pope_yes(strip_think(r['model_answer'])), r['response'] == 'yes'
        per.append({'question_id': r['question_id'], 'pred_yes': pred, 'correct': pred == lab, 'source': 'rule', 'split': r['category']})
        c = by[r['category']]
        c[0 if pred and lab else 1 if pred else 2 if not lab else 3] += 1
    out = {}
    for sp, (tp, fp, tn, fn) in sorted(by.items()):
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[sp] = {'acc': 100 * (tp + tn) / (tp + fp + tn + fn), 'precision': 100 * p, 'recall': 100 * rc,
                   'f1': 100 * (2 * p * rc / (p + rc) if p + rc else 0.0), 'yes_ratio': 100 * (tp + fp) / (tp + fp + tn + fn)}
    out['score'] = _mean([v['acc'] for k, v in out.items() if isinstance(v, dict)])
    out['f1'] = _mean([v['f1'] for k, v in out.items() if isinstance(v, dict)])
    return per, out


HB_JUDGE_HEAD = ('Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. '
                 'If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. If the prediction answer is unclear about the answer, please generate "unclear". \n\n Question:')


def hb_rule(text):
    t = md_clean(strip_think(text)).lower()
    segs = [t]
    ms = list(re.finditer(r'final\s+answer\s*[:：]?|(?<![a-z])answer\s*[:：]|答案\s*[:：]?', t))
    if ms:
        segs.insert(0, t[ms[-1].end():])
    for seg in segs:
        m = re.match(r'\s*([a-z]+)', seg)
        if m and m.group(1) in ('yes', 'no'):
            return m.group(1)
    return None


def score_hallusionbench(rows, judge=None):
    per, ask = [], []
    for r in rows:
        x = {k: r.get(k) for k in ('category', 'subcategory', 'set_id', 'figure_id', 'question_id', 'visual_input', 'gt_answer')}
        x['orig_question_id'], x['question_id'] = r['orig_question_id'], r['orig_question_id']
        x['qid'] = r['question_id']
        pred = hb_rule(r['model_answer'])
        if pred is not None:
            x['corr'] = '1' if (pred == 'yes') == (str(r['gt_answer']) == '1') else '0'
            x['source'] = 'rule'
        else:
            x['corr'] = None
            ask.append((x, r))
        per.append(x)
    outs = judge.map('hallusionbench', ask, lambda xr: user(HB_JUDGE_HEAD + xr[1]['query'] + '\nReference answer: ' + str(xr[1]['gt_answer_details'])
                                                         + '\nPrediction answer:' + strip_think(xr[1]['model_answer']) + '\nOutput:')) if (judge and ask) else [None] * len(ask)
    for (x, _), o in zip(ask, outs):
        if o is None:
            x['corr'], x['source'] = '0', 'judge_failed'
        else:
            x['corr'] = '0' if 'incorrect' in o.lower() else '1' if 'correct' in o.lower() else '2'
            x['source'] = 'judge'
    for x in per:
        if x["category"] == "VS" and int(x["figure_id"]) == 0:
            x["correct"] = 1 if int(x['corr']) == 1 or int(x['corr']) == 2 else 0
        else:
            x["correct"] = 1 if int(x['corr']) == 1 else 0
    a_all = _hb_eval_all(per, 'corr')
    a_fig = _hb_eval_fig(per)
    a_pair = _hb_eval_pair_all(per)
    for x in per:
        x['question_id'] = x.pop('qid')
        x['correct'] = bool(x['correct'])
    return per, {'score': 100 * a_all['correct'] / a_all['total'], 'aAcc': 100 * a_all['correct'] / a_all['total'],
                 'fAcc': 100 * a_fig['correct'] / a_fig['total'], 'qAcc': 100 * a_pair['correct'] / a_pair['total'],
                 'n_questions': a_all['total'], 'n_figures': a_fig['total'], 'n_pairs': a_pair['total']}


def _hb_eval_fig(data):
    eval_fig_dict = dict()
    for r in data:
        if r["category"] == "VS" and str(r["figure_id"]) == "0":
            continue
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"])])
        if name in eval_fig_dict:
            c, t = eval_fig_dict[name]
            eval_fig_dict[name] = (c + r["correct"], t + 1)
        else:
            eval_fig_dict[name] = (r["correct"], 1)
    return {'total': len(eval_fig_dict), 'correct': sum(1 for v in eval_fig_dict.values() if v[0] == v[1])}


def _hb_eval_all(data, model_correctness_entry):
    eval_all_dict = dict()
    for r in data:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"]), str(r["question_id"])])
        assert name not in eval_all_dict
        eval_all_dict[name] = r["correct"]
    return {'total': len(eval_all_dict), 'correct': sum(1 for v in eval_all_dict.values() if v)}


def _hb_eval_pair_all(data):
    d = dict()
    for r in data:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        c, t = d.get(name, (0, 0))
        d[name] = (c + r["correct"], t + 1)
    return {'total': len(d), 'correct': sum(1 for c, t in d.values() if c == t)}


def cambrian_mmvp_extract(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    text = text.strip().rstrip('.:,').lstrip('(').rstrip(')')
    return text


MMVP_SYSTEM = 'You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.'


def score_mmvp(rows, judge=None):
    per, ask = [], []
    for r in rows:
        opts = [o.strip().lower() for o in re.split(r'\([a-z]\)', r['options']) if o.strip()]
        clean = md_clean(strip_think(r['model_answer']))
        ans = cambrian_mmvp_extract(clean)
        gt = r['response']
        for cand in (ans.lstrip(':：. \n'), (marker_answer(clean) or '').lower()):
            m = re.match(r'^\(?([ab])(?:\)|[.:]|$)', cand.strip())
            if m:
                ans = m.group(1)
                break
        x = {'question_id': r['question_id'], 'pair_id': r['pair_id'], 'extracted': ans}
        if ans in ('a', 'b') or ans in opts:
            x['correct'] = (ans == gt) or (gt in 'ab' and len(opts) > ord(gt) - 97 and ans == opts[ord(gt) - 97])
            x['source'] = 'rule'
        else:
            ask.append((x, r))
        per.append(x)
    outs = judge.map('mmvp', ask, lambda xr: [{'role': 'system', 'content': MMVP_SYSTEM},
                                               {'role': 'user', 'content': f"Given the following question {xr[1]['query']}, the correct answer is {xr[1]['answer_raw']}. "
                                                                           f"Does the following answer correctly answers the question, answer:{strip_think(xr[1]['model_answer'])}?"}]) if (judge and ask) else [None] * len(ask)
    for (x, _), o in zip(ask, outs):
        if o is None:
            x['correct'], x['source'] = False, 'judge_failed'
        else:
            x['correct'], x['source'] = bool(re.match(r'^(yes|no)$', o.strip(), re.I)) and o.strip().lower() == 'yes', 'judge'
    pairs = defaultdict(list)
    for x in per:
        pairs[x['pair_id']].append(x['correct'])
    return per, {'score': 100 * _mean([all(v) for v in pairs.values()]), 'question_acc': 100 * _mean([x['correct'] for x in per]), 'n_pairs': len(pairs)}


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mathvista_normalize(extraction, choices, question_type, answer_type, precision):
    if question_type == 'multi_choice':
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()
        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            distances = [levenshtein(extraction, choice) for choice in choices]
            normalized_extraction = choices[distances.index(min(distances))]
        assert normalized_extraction in choices
    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None
    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None
    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None
    return normalized_extraction


def mathvista_rule(r):
    response = strip_think(r['model_answer'])
    if response == "":
        return None
    choices = r.get('choices') or []
    if r['question_type'] == 'multi_choice' and response in choices:
        return response
    if r['answer_type'] == 'integer':
        try:
            return str(int(response))
        except Exception:
            pass
    if r['answer_type'] == 'float':
        try:
            return str(float(response))
        except Exception:
            pass
    m = marker_answer(response)
    if m is None:
        return None
    if r['question_type'] == 'multi_choice':
        L = option_letter(m, ''.join(chr(65 + i) for i in range(len(choices))))
        if L:
            return L
        return m if m in choices else None
    if r['answer_type'] in ('integer', 'float'):
        return first_number(m)
    if r['answer_type'] == 'list':
        return None
    return m


def score_mathvista(rows, judge=None):
    from official.mathvista_prompts import demo_prompt
    per, ask = [], []
    for r in rows:
        x = {'question_id': r['question_id']}
        e = mathvista_rule(r)
        if e is not None:
            x['extraction'], x['source'] = e, 'rule'
        else:
            ask.append((x, r))
        per.append((x, r))
    outs = judge.map('mathvista', ask, lambda xr: user(f"{demo_prompt.strip()}\n\n{xr[1]['query']}\n\n{strip_think(xr[1]['model_answer'])}\n\nExtracted answer: ")) if (judge and ask) else [None] * len(ask)
    for (x, _), o in zip(ask, outs):
        x['extraction'], x['source'] = (o, 'judge') if o is not None else ('', 'judge_failed')
    res = []
    for x, r in per:
        try:
            norm = mathvista_normalize(x['extraction'], r.get('choices') or [], r['question_type'], r['answer_type'], r.get('precision') or 0)
        except Exception:
            norm = None
        x['normalized'] = norm
        x['correct'] = (norm == r['answer'])
        x['category'] = r.get('category')
        res.append(x)
    return res, {'score': 100 * _mean([x['correct'] for x in res])}


def score_mathverse(rows, judge=None):
    from official.mathverse_prompts import demo_prompt_extract, demo_prompt_score
    per, need_s1, need_s2 = [], [], []
    for r in rows:
        x = {'question_id': r['question_id'], 'version': r['problem_version']}
        resp = strip_think(r['model_answer'])
        m = marker_answer(resp)
        if r['question_type'] == 'multi-choice':
            L = option_letter(m) if m is not None else None
            if L:
                x.update(extraction=L, correct=(L == str(r['answer']).strip()), source='rule')
            else:
                need_s1.append((x, r))
        else:
            n, g = first_number(m) if m is not None else None, first_number(str(r['answer']))
            if m is not None and n is not None and g is not None and re.fullmatch(r'-?[\d,]*\.?\d+', str(r['answer']).strip().replace(',', '')) \
                    and abs(float(n) - float(g)) <= 1e-6 * max(1.0, abs(float(g))):
                x.update(extraction=m, correct=True, source='rule')
            elif m is not None:
                x['extraction'] = m
                need_s2.append((x, r))
            else:
                need_s1.append((x, r))
        per.append(x)
    outs = judge.map('mathverse_s1', need_s1, lambda xr: user(f"{demo_prompt_extract.strip()}\n\nModel response: '{strip_think(xr[1]['model_answer'])}'\nExtracted Answer: ")) if (judge and need_s1) else [None] * len(need_s1)
    for (x, r), o in zip(need_s1, outs):
        if o is None:
            x.update(extraction='', correct=False, source='judge_failed')
            continue
        x['extraction'] = o.strip()
        x['source'] = 'judge'
        if r['question_type'] == 'multi-choice' and option_letter(x['extraction']):
            x['correct'] = option_letter(x['extraction']) == str(r['answer']).strip()
        else:
            need_s2.append((x, r))
    outs = judge.map('mathverse_s2', need_s2, lambda xr: user(demo_prompt_score.strip().format(question=xr[1]['question'], gt=xr[1]['answer'], extraction=xr[0]['extraction']))) if (judge and need_s2) else [None] * len(need_s2)
    for (x, r), o in zip(need_s2, outs):
        if o is None:
            x.update(correct=False, source='judge_failed')
            continue
        v = o.replace("Judgement:", "").strip()
        x['correct'] = v.startswith('1')
        x['source'] = 'judge'
    vers = sorted({x['version'] for x in per})
    return per, {'score': 100 * _mean([x['correct'] for x in per]), 'by_version': {v: 100 * _mean([x['correct'] for x in per if x['version'] == v]) for v in vers}}


def _mean(v):
    v = list(v)
    return float(sum(v)) / len(v) if v else float('nan')


SCORERS = {'mmstar': score_mmstar, 'cvbench': score_cvbench, 'mmvp': score_mmvp, 'pope': score_pope, 'hallusionbench': score_hallusionbench,
           'mathvista': score_mathvista, 'mathverse': score_mathverse, 'mathvision': score_mathvision, 'wemath': score_wemath, 'dynamath': score_dynamath}


def judge_stats(per):
    c = Counter(p.get('source', 'rule') for p in per)
    n = max(len(per), 1)
    return {'n': len(per), 'rule': c.get('rule', 0), 'judge': c.get('judge', 0), 'judge_failed': c.get('judge_failed', 0),
            'judge_rate': 100 * (c.get('judge', 0) + c.get('judge_failed', 0)) / n}
