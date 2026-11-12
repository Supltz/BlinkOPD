import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from select_k500 import read_pool
CUE = "**Answer: ("
CUTS = (0, 25, 50, 75, 100)
N_ROLL = 4
GEN_SEED = 20261004
MAX_NEW_TOKENS = 1024
TOPK = 100
IMG_REFS = ("N", "black", "noise", "mismatch")
NOISE_MEAN, NOISE_STD = 128.0, 64.0
VAD_SCALE = 0.10
VAD = {"cap": 0.8, "ridge": 1e-3, "step": 1.0, "max_shift": 20.0}
CV = {"max_iou": 0.1, "min_scale": 0.05, "max_scale": 0.5, "seed": 20260813, "step": 1}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def mean_color(img):
    import numpy as np
    from PIL import Image
    fill = tuple(np.rint(np.asarray(img, dtype=np.float32).mean(axis=(0, 1))).clip(0, 255).astype(np.uint8).tolist())
    return Image.new("RGB", img.size, fill)


def null_images(img, seed, other):
    import numpy as np
    from PIL import Image
    w, h = img.size
    rng = np.random.default_rng(int(seed))
    noise = Image.fromarray(np.clip(np.rint(rng.normal(NOISE_MEAN, NOISE_STD, (h, w, 3))), 0, 255).astype(np.uint8), "RGB")
    return {"N": mean_color(img), "black": Image.new("RGB", (w, h), (0, 0, 0)), "noise": noise,
            "mismatch": other.resize((w, h), Image.BICUBIC)}


def degrade(img):
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    low = img.resize((max(1, round(w * VAD_SCALE)), max(1, round(h * VAD_SCALE))), Image.Resampling.BILINEAR)
    return low.resize((w, h), Image.Resampling.NEAREST)


def masked_crop(img):
    import numpy as np
    from PIL import ImageDraw
    output = img.convert("RGB").copy()
    rng = np.random.default_rng(0)
    width, height = output.size
    area = max(1, int(width * height / 3.0))
    aspect = float(rng.uniform(0.5, 2.0))
    w = min(width, max(1, int((area * aspect) ** 0.5)))
    h = min(height, max(1, int(area / w)))
    x = int(rng.integers(0, max(1, width - w + 1)))
    y = int(rng.integers(0, max(1, height - h + 1)))
    ImageDraw.Draw(output).rectangle((x, y, x + w, y + h), fill=(0, 0, 0))
    return output


def crop_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, a[2] - a[0]) * max(1.0, a[3] - a[1])
    area_b = max(1.0, b[2] - b[0]) * max(1.0, b[3] - b[1])
    return inter / max(1.0, area_a + area_b - inter)


def neg_seed(set_name, row):
    mat = f"{CV['seed']}:{CV['step']}:{set_name}-{row}:sample_shared:0"
    return int.from_bytes(hashlib.sha256(mat.encode("utf-8")).digest()[:8], "big")


def negative_crop(full_path, box, output_size, seed):
    from PIL import Image
    image = Image.open(full_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    positive = (max(0.0, min(x1, width - 1)), max(0.0, min(y1, height - 1)), max(1.0, min(x2, width)), max(1.0, min(y2, height)))
    rng = random.Random(int(seed))
    cands = []
    for _ in range(64):
        area = width * height * rng.uniform(CV["min_scale"], CV["max_scale"])
        aspect = 2.0 ** rng.uniform(-1.0, 1.0)
        cw = max(2.0, min(width, (area * aspect) ** 0.5))
        ch = max(2.0, min(height, area / cw))
        left = rng.uniform(0.0, max(0.0, width - cw))
        top = rng.uniform(0.0, max(0.0, height - ch))
        cand = (left, top, left + cw, top + ch)
        iou = crop_iou(cand, positive)
        cands.append((iou, cand))
        if iou <= CV["max_iou"]:
            return image.crop(tuple(int(round(v)) for v in cand)).resize(output_size, Image.Resampling.BICUBIC), iou
    iou, cand = min(cands, key=lambda it: it[0])
    if iou > CV["max_iou"]:
        raise ValueError(f"no negative crop with IoU <= {CV['max_iou']}: best {iou:.4f}")
    return image.crop(tuple(int(round(v)) for v in cand)).resize(output_size, Image.Resampling.BICUBIC), iou


def options_of(question):
    return sorted(set(re.findall(r"(?m)^\s*\(?([A-H])[\.\)]\s", question))) or list("ABCD")


def letter_tokens(tok):
    found = {}
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lstrip("(").strip()
        if len(s) == 1 and s in "ABCDEFGH":
            found.setdefault(s, []).append(i)
    return found


def option_dist(probs, letters, opts):
    raw = {o: float(probs[letters[o]].sum()) for o in opts}
    mass = sum(raw.values())
    return {o: v / mass for o, v in raw.items()} if mass > 0 else {o: 1 / len(opts) for o in opts}


def support_lp(lp, idx):
    import torch
    g = lp.gather(1, idx)
    s = torch.logsumexp(g, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat((g, torch.log(-torch.expm1(s))), 1)


def jsd(a, b):
    import torch
    m = torch.logaddexp(a, b) - math.log(2.0)
    return 0.5 * ((a.exp() * (a - m)).sum(-1) + (b.exp() * (b - m)).sum(-1))


def vad_target(S, T, D):
    import torch
    center = lambda x: x - x.mean(dim=-1, keepdim=True)
    phi_s, phi_plus, phi_minus = center(S), center(T), center(D)
    r = phi_plus - phi_s
    u = phi_plus - phi_minus
    dot_ru = (r * u).sum(dim=-1, keepdim=True)
    beta = torch.clamp(dot_ru, min=0.0) / ((u * u).sum(dim=-1, keepdim=True) + VAD["ridge"])
    r_vis_single = beta * u
    u_pos = torch.clamp(u, min=0.0)
    u_neg = torch.clamp(u, max=0.0)
    s_pos = torch.clamp((r * u_pos).sum(dim=-1, keepdim=True), min=0.0)
    s_neg = torch.clamp((r * u_neg).sum(dim=-1, keepdim=True), min=0.0)
    s_total = s_pos + s_neg + 1e-8
    alpha_pos = torch.minimum(s_pos / s_total, torch.full_like(s_pos, VAD["cap"]))
    alpha_neg = s_neg / s_total
    budget = torch.linalg.vector_norm(r_vis_single, dim=-1, keepdim=True)
    beta_pos = (alpha_pos * budget) / (torch.linalg.vector_norm(u_pos, dim=-1, keepdim=True) + 1e-8)
    beta_neg = (alpha_neg * budget) / (torch.linalg.vector_norm(u_neg, dim=-1, keepdim=True) + 1e-8)
    r_vis = beta_pos * u_pos + beta_neg * u_neg
    shift = torch.clamp(VAD["step"] * r_vis, min=-VAD["max_shift"], max=VAD["max_shift"])
    return torch.log_softmax(phi_s + shift, dim=-1)


def opt_from_support(lq, idx, letters, opts):
    import torch
    p = lq[:TOPK].exp()
    raw = {o: float(p[torch.isin(idx, letters[o])].sum()) for o in opts}
    mass = sum(raw.values())
    return {o: v / mass for o, v in raw.items()} if mass > 0 else {o: 1 / len(opts) for o in opts}


def response_ids(tok, text, length, im_end):
    if text.endswith("\n"):
        body = tok.encode(text[:-1], add_special_tokens=False)
        if len(body) + 2 == length:
            return body + [im_end]
    return tok.encode(text, add_special_tokens=False)


def top(d):
    return max(d, key=d.get)


def r6(x):
    return [round(float(v), 6) for v in x.tolist()]


class Model:
    def __init__(self, path, template):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        cfg = self.model.config
        self.image_token = cfg.image_token_id
        self.visual = {cfg.image_token_id, cfg.video_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id}
        self.template = Path(template).read_text()
        self.eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prompt(self, image, text, student=False, copies=1):
        if student:
            from qwen_vl_utils import fetch_image
            image = fetch_image({"image": image}, image_patch_size=self.processor.image_processor.patch_size)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text.replace("<image>", "")}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=self.template)
        batch = self.processor(text=[rendered] * copies, images=[image] * copies, return_tensors="pt")
        return {k: v.to("cuda:0") for k, v in batch.items()}

    def generate(self, prompt, seed):
        torch = self.torch
        from transformers import GenerationConfig
        gc = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW_TOKENS,
                              eos_token_id=[self.eos], pad_token_id=self.eos)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = self.model.generate(**prompt, generation_config=gc)
        rolls = []
        for row in out[:, prompt["input_ids"].shape[1]:].tolist():
            cut = next((i for i, t in enumerate(row) if t == self.eos), len(row))
            ids = row[:cut]
            if any(t in self.visual for t in ids):
                raise ValueError("rollout contains a visual special token")
            rolls.append({"text": self.tok.decode(ids, skip_special_tokens=True), "len": len(ids)})
        return rolls

    def _extend(self, prompt, suffix):
        torch = self.torch
        ids = torch.as_tensor(suffix, device="cuda:0", dtype=torch.long)[None]
        batch = dict(prompt)
        batch["input_ids"] = torch.cat((prompt["input_ids"], ids), 1)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        if "mm_token_type_ids" in batch:
            batch["mm_token_type_ids"] = torch.cat((prompt["mm_token_type_ids"], torch.zeros_like(ids)), 1)
        return batch

    def last_logits(self, prompt, suffix):
        with self.torch.no_grad():
            out = self.model(**self._extend(prompt, suffix), logits_to_keep=1, use_cache=False)
        return out.logits[0, -1].float()

    def seq_logits(self, prompt, response):
        torch = self.torch
        start = prompt["input_ids"].shape[1]
        pos = torch.arange(start - 1, start + len(response) - 1, device="cuda:0")
        with torch.no_grad():
            out = self.model(**self._extend(prompt, response), logits_to_keep=pos, use_cache=False)
        return out.logits[0].float()

    def cut_prompt(self, PT):
        ids = PT["input_ids"]
        pos, _ = self.model.model.get_rope_index(ids, mm_token_type_ids=PT["mm_token_type_ids"], image_grid_thw=PT["image_grid_thw"],
                                                 attention_mask=PT["attention_mask"])
        keep = ids[0] != self.image_token
        return {"input_ids": ids[:, keep], "pos": pos[:, :, keep]}

    def txt_prompt(self, text):
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "text", "text": text.replace("<image>", "")}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=self.template)
        ids = self.tok(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda:0")
        if any(int(t) in self.visual for t in ids[0]):
            raise ValueError("text-only prompt contains a visual special token")
        pos = torch.arange(ids.shape[1], device="cuda:0").view(1, 1, -1).expand(3, 1, -1).clone()
        return {"input_ids": ids, "pos": pos}

    def x_logits(self, xp, suffix):
        torch = self.torch
        ids = torch.as_tensor(suffix, device="cuda:0", dtype=torch.long)[None]
        ext = xp["pos"][:, :, -1:] + torch.arange(1, ids.shape[1] + 1, device="cuda:0").view(1, 1, -1)
        full = torch.cat((xp["input_ids"], ids), 1)
        batch = {"input_ids": full, "attention_mask": torch.ones_like(full), "position_ids": torch.cat((xp["pos"], ext), 2)}
        with torch.no_grad():
            out = self.model(**batch, logits_to_keep=1, use_cache=False)
        return out.logits[0, -1].float()


def read_items(path):
    with open(path, newline="") as f:
        return [{"row": int(r["row"]), "set": r["set"], "teacher_image": r["teacher_image"], "mismatch_row": int(r["mismatch_row"])} for r in csv.DictReader(f)]


def read_mmerw(path):
    with open(path, newline="") as f:
        return [{"pos": int(r["pos"]), "index": int(r["index"]), "mismatch_index": int(r["mismatch_index"])} for r in csv.DictReader(f)]


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def done_keys(parts_dir, prefix, key):
    seen = set()
    for p in sorted(parts_dir.glob(prefix + "*.jsonl")):
        for r in read_jsonl(p):
            seen.add(key(r))
    return seen


def append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def check_same_vision(P, ref, names):
    for c in names:
        for key in ("input_ids", "mm_token_type_ids", "image_grid_thw"):
            if key in P[ref] and not P[c][key].equal(P[ref][key]):
                raise ValueError(f"{c}: {key} differs from {ref}")


def probe_item(m, letters, data_dir, it, jl, rolls, set_name):
    torch = m.torch
    tok = m.tok
    r = it["row"]
    row = jl[r]
    if Path(row["teacher_images"][0]).name != it["teacher_image"]:
        raise ValueError(f"row {r}: teacher image differs from items.csv")
    text, q = row["problem"], row["extra_info"]["question"]
    gt, opts = str(row["answer"]), options_of(q)
    cue0 = tok.encode(CUE, add_special_tokens=False)
    cue1 = tok.encode("\n\n" + CUE, add_special_tokens=False)
    crop = load_rgb(data_dir / row["teacher_images"][0])
    other = load_rgb(data_dir / jl[it["mismatch_row"]]["teacher_images"][0])
    P = {"S": m.prompt(load_rgb(data_dir / row["images"][0]), text, student=True), "T": m.prompt(crop, text)}
    for c, im in null_images(crop, r, other).items():
        P[c] = m.prompt(im, text)
    P["D"] = m.prompt(degrade(crop), text)
    P["M"] = m.prompt(masked_crop(crop), text)
    neg, iou = negative_crop(data_dir / row["images"][0], row["bbox"], crop.size, neg_seed(set_name, r))
    P["NEG"] = m.prompt(neg, text)
    check_same_vision(P, "T", IMG_REFS + ("D", "NEG", "M"))
    XP = {"cut": m.cut_prompt(P["T"]), "txt": m.txt_prompt(text)}
    letter_ids = {o: [int(i) for i in letters[o].tolist()] for o in opts}
    out_rolls = []
    for k, ro in enumerate(rolls):
        ids = response_ids(tok, ro["text"], ro["len"], m.im_end)
        L = len(ids)
        cuts = [round(a / 100 * L) for a in CUTS]
        tok_jsd = []
        if L:
            ls = m.seq_logits(P["S"], ids).log_softmax(-1)
            lt = m.seq_logits(P["T"], ids).log_softmax(-1)
            tok_jsd = r6(jsd(ls, lt))
            del ls, lt
        probes = {}
        for a, n in zip(CUTS, cuts):
            suf = ids[:n] + (cue1 if n else cue0)
            logit = {c: m.last_logits(P[c], suf) for c in ("S", "T") + IMG_REFS + ("D", "NEG", "M")}
            for v in ("cut", "txt"):
                logit[v] = m.x_logits(XP[v], suf)
            pr = {c: option_dist(logit[c].softmax(-1), letters, opts) for c in ("S", "T") + IMG_REFS + ("cut", "txt")}
            lp = {c: logit[c].log_softmax(-1) for c in ("S", "T", "D", "NEG", "M")}
            idx = lp["S"].topk(TOPK).indices
            sup = {c: support_lp(lp[c][None], idx[None])[0] for c in lp}
            pr["vad"] = opt_from_support(vad_target(sup["S"], sup["T"], sup["D"]), idx, letters, opts)
            pr["cv"] = {"idx": idx.tolist(), "S": r6(sup["S"]), "T": r6(sup["T"]), "N": r6(sup["NEG"]),
                        "jsd_ST": round(float(jsd(lp["S"], lp["T"])), 6)}
            pr["opdv"] = {o: float(lp["T"][j] - lp["M"][j])
                          for o in opts for j in [letters[o][lp["S"][letters[o]].argmax()]]}
            probes[str(a)] = pr
        out_rolls.append({"k": k, "len": L, "len_gen": ro["len"], "chosen": top(probes["100"]["S"]),
                          "tok_jsd_ST": tok_jsd, "probes": probes})
        torch.cuda.empty_cache()
    return {"row": r, "set": set_name, "gt": gt, "options": opts, "letter_ids": letter_ids,
            "neg_iou": round(iou, 4), "rollouts": out_rolls}


def etoken_item(m, letters, eval_dir, it, by_index):
    torch = m.torch
    rec = by_index[it["index"]]
    img = load_rgb(eval_dir / rec["images"][0])
    other = load_rgb(eval_dir / by_index[it["mismatch_index"]]["images"][0])
    Lt = torch.tensor(sorted({i for v in letters.values() for i in v.tolist()}), device="cuda:0")
    Et = letters["E"]
    masses = lambda lp: (lambda p: {"letter": float(p[Lt].sum()), "E": float(p[Et].sum())})(lp.softmax(-1))
    PS = m.prompt(img, rec["query"])
    lp = {"real": m.last_logits(PS, []).softmax(-1).clamp(min=1e-30).log()}
    for c, im in null_images(img, it["pos"], other).items():
        lp[c] = m.last_logits(m.prompt(im, rec["query"]), []).softmax(-1).clamp(min=1e-30).log()
    lp["cut"] = m.x_logits(m.cut_prompt(PS), []).softmax(-1).clamp(min=1e-30).log()
    lp["txt"] = m.x_logits(m.txt_prompt(rec["query"]), []).softmax(-1).clamp(min=1e-30).log()
    return {"pos": it["pos"], "index": it["index"], "gt": rec["response"], **{c: masses(x) for c, x in lp.items()}}


def worker(a):
    import torch
    out = Path(a.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tag = f"{a.shard:02d}of{a.num_shards:02d}"
    data_dir = Path(a.data_dir)
    jl = read_pool(data_dir / "train.jsonl")
    items = read_items(a.items)
    mine = [it for i, it in enumerate(items) if i % a.num_shards == a.shard]
    given = {}
    if a.rollouts:
        for r in read_jsonl(a.rollouts):
            given.setdefault((r["set"], r["row"]), {})[r["k"]] = r
    gen_done = {}
    for p in sorted(parts.glob("rollouts_*.jsonl")):
        for r in read_jsonl(p):
            gen_done.setdefault((r["set"], r["row"]), {})[r["k"]] = r
    done = done_keys(parts, "readout_", lambda r: (r["set"], r["row"]))
    todo = [it for it in mine if (it["set"], it["row"]) not in done]
    et_todo = []
    if not a.skip_etoken:
        et_items = [it for i, it in enumerate(read_mmerw(a.mmerw)) if i % a.num_shards == a.shard]
        et_done = done_keys(parts, "etoken_", lambda r: r["pos"])
        et_todo = [it for it in et_items if it["pos"] not in et_done]
    log(f"shard {tag}: {len(todo)} probe questions and {len(et_todo)} MME-RealWorld-CN questions to do")
    if not todo and not et_todo:
        return
    t0 = time.time()
    m = Model(a.model, a.template)
    letters = {o: torch.tensor(v, device="cuda:0") for o, v in letter_tokens(m.tok).items()}
    log(f"shard {tag}: model loaded in {time.time() - t0:.0f} s")
    t0 = time.time()
    for j, it in enumerate(todo):
        key = (it["set"], it["row"])
        if key in given:
            rolls = [given[key][k] for k in range(N_ROLL)]
        elif key in gen_done and len(gen_done[key]) == N_ROLL:
            rolls = [gen_done[key][k] for k in range(N_ROLL)]
        else:
            row = jl[it["row"]]
            prompt = m.prompt(load_rgb(data_dir / row["images"][0]), row["problem"], student=True, copies=N_ROLL)
            rolls = m.generate(prompt, seed=GEN_SEED + it["row"])
            for k, ro in enumerate(rolls):
                append(parts / f"rollouts_{tag}.jsonl", {"set": it["set"], "row": it["row"], "k": k, "len": ro["len"], "text": ro["text"]})
        append(parts / f"readout_{tag}.jsonl", probe_item(m, letters, data_dir, it, jl, rolls, it["set"]))
        if (j + 1) % 8 == 0 or j + 1 == len(todo):
            rate = (time.time() - t0) / (j + 1)
            log(f"shard {tag}: probe {j + 1}/{len(todo)} questions, {rate:.0f} s per question, about {rate * (len(todo) - j - 1) / 60:.0f} min left")
    if et_todo:
        eval_dir = Path(a.eval_data_dir)
        by_index = {int(r["index"]): r for r in json.loads((eval_dir / "MME_RealWorld_CN.json").read_text(encoding="utf-8"))}
        for j, it in enumerate(et_todo):
            append(parts / f"etoken_{tag}.jsonl", etoken_item(m, letters, eval_dir, it, by_index))
        log(f"shard {tag}: MME-RealWorld-CN read-outs done ({len(et_todo)} questions)")
    log(f"shard {tag}: finished")


def merge(a):
    out = Path(a.out)
    parts = out / "parts"
    items = read_items(a.items)
    order = {(it["set"], it["row"]): i for i, it in enumerate(items)}
    recs = {}
    for p in sorted(parts.glob("readout_*.jsonl")):
        for r in read_jsonl(p):
            recs.setdefault((r["set"], r["row"]), r)
    missing = [k for k in order if k not in recs]
    if missing:
        sys.exit(f"{len(missing)} probe questions have no read-outs yet (first: {missing[0]}); rerun the same command to resume")
    with gzip.open(out / "readouts.jsonl.gz", "wt", encoding="utf-8") as f:
        for k in sorted(order, key=order.get):
            f.write(json.dumps(recs[k], ensure_ascii=False) + "\n")
    rolls = {}
    for p in sorted(parts.glob("rollouts_*.jsonl")):
        for r in read_jsonl(p):
            rolls.setdefault((r["set"], r["row"], r["k"]), r)
    if rolls:
        with gzip.open(out / "rollouts.jsonl.gz", "wt", encoding="utf-8") as f:
            for k in sorted(rolls, key=lambda x: (order.get(x[:2], 10 ** 9), x[2])):
                f.write(json.dumps(rolls[k], ensure_ascii=False) + "\n")
    log(f"merged {len(recs)} probe questions into {out / 'readouts.jsonl.gz'}")
    if not a.skip_etoken:
        want = [it["pos"] for it in read_mmerw(a.mmerw)]
        et = {}
        for p in sorted(parts.glob("etoken_*.jsonl")):
            for r in read_jsonl(p):
                et.setdefault(r["pos"], r)
        miss = [x for x in want if x not in et]
        if miss:
            sys.exit(f"{len(miss)} MME-RealWorld-CN questions have no read-outs yet; rerun the same command to resume")
        with open(out / "etoken.jsonl", "w", encoding="utf-8") as f:
            for x in want:
                f.write(json.dumps(et[x], ensure_ascii=False) + "\n")
        log(f"merged {len(want)} MME-RealWorld-CN questions into {out / 'etoken.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--template", default=str(ROOT / "chat_templates" / "perception_chat_template_qwen35.jinja"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "vision_opd_6k"))
    ap.add_argument("--eval-data-dir", default=str(ROOT / "data" / "eval"))
    ap.add_argument("--items", default=str(HERE / "items.csv"))
    ap.add_argument("--mmerw", default=str(HERE / "mmerw_cn_300.csv"))
    ap.add_argument("--rollouts", default="")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "probe"))
    ap.add_argument("--skip-etoken", action="store_true")
    a = ap.parse_args()
    worker(a) if a.cmd == "worker" else merge(a)


if __name__ == "__main__":
    main()
