import argparse
import csv
import json
import os
from pathlib import Path

from select_k500 import ROOT, read_pool, validate

def image_path(pool, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Expected a relative dataset image path: {relative}")
    path = pool / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


def context_crop(source, box, target):
    from PIL import Image
    with Image.open(source) as original:
        image = original.convert("RGB")
    width, height = image.size
    def axis(a, b, size):
        n = min(size, max(1, round((b - a) * 3.0)))
        left = min(max(round((a + b) / 2 - n / 2), 0), size - n)
        return left, left + n
    left, right = axis(box[0], box[2], width)
    top, bottom = axis(box[1], box[3], height)
    w, h = right - left, bottom - top
    scale = min(2.0, (width * height / (w * h)) ** 0.5)
    result = image.crop((left, top, right, bottom)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target)


def training_record(row, pool, teacher, glance=None):
    extra = {"answer": row["answer"], "question": row["extra_info"]["question"],
             "source_extra_info": row["extra_info"]}
    if glance is not None:
        extra.update(vg_first_letter=glance, vg_options="ABCD")
    return {"data_source": "zwz_rl_vqa_bbox_teacher", "prompt": [{"role": "user", "content": row["problem"]}],
            "images": [{"path": image_path(pool, row["images"][0])}], "bbox_images": [{"path": str(teacher)}],
            "ability": "visual_question_answering", "reward_model": {"ground_truth": row["answer"], "style": "none"},
            "extra_info": extra}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=str(ROOT / "data" / "vision_opd_6k"))
    p.add_argument("--out", default=str(ROOT / "data" / "blinkopd_k500"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--baseline-out")
    a = p.parse_args()
    pool, out = Path(a.pool).resolve(), Path(a.out).resolve()
    if not (pool / "train.jsonl").exists():
        raise FileNotFoundError("Download and extract Vision-OPD-6K from the upstream link in README, then pass --pool /path/to/Vision-OPD-6K")
    rows = read_pool(pool / "train.jsonl")
    for row in rows:
        image_path(pool, row["images"][0])
        image_path(pool, row["teacher_images"][0])
    if a.baseline_out:
        destination = Path(a.baseline_out).resolve()
        if destination == pool:
            raise ValueError("Baseline output must differ from the source pool")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "train.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        if not target.exists() or target.read_bytes() != content:
            temporary = target.with_suffix(".partial.jsonl")
            temporary.write_bytes(content)
            os.replace(temporary, target)
        print(f"Prepared {len(rows)} decontaminated baseline rows in {target}")
        return
    if a.check_only:
        print(f"Verified {len(rows)} pool rows in {pool}")
        return
    selected, training = validate(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    with (ROOT / "indices" / "first_glance.csv").open(newline="") as f:
        glances = {int(r["row"]): r for r in csv.DictReader(f)}
    if set(glances) != set(selected):
        raise ValueError("First-glance index differs from the selected image index")
    out.mkdir(parents=True, exist_ok=True)
    for size in ("4b", "9b"):
        by_row = {}
        for i in selected:
            row = rows[i]
            teacher = Path(image_path(pool, row["teacher_images"][0]))
            if size == "9b":
                teacher = out / "teacher_ctx3" / teacher.name
                if not teacher.exists():
                    context_crop(image_path(pool, row["images"][0]), row["bbox"], teacher)
            by_row[i] = training_record(row, pool, teacher, glances[i][size])
        path = out / f"train_{size}.parquet"
        temporary = path.with_suffix(".partial.parquet")
        pq.write_table(pa.Table.from_pylist([by_row[i] for i in training]), temporary)
        os.replace(temporary, path)
        print(f"Wrote {path}: {len(selected)} questions, {len(training)} training rows")
    (out / "selection.json").write_text(json.dumps({"pool_rows": len(rows), "unique_questions": 500,
        "training_rows": len(training), "subset_seed": 42, "row_shuffle_seed": 44,
        "teacher_4b": "original tight crop", "teacher_9b": "3x bounding-box context crop",
        "first_glance": "released frozen-base-model readouts on original tight crops"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
