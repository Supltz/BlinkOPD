import argparse
import csv
import hashlib
import json
import os
import tarfile
from pathlib import Path

from select_k500 import ROOT, read_pool, validate

DATASET = "Danny-Lian/Vision-OPD-6K-Decontaminated"
DATASET_REVISION = "5ceec8da2b2c2192c2c4b64626beb1a09240c834"


def download_pool(destination, revision):
    from huggingface_hub import snapshot_download
    source = Path(snapshot_download(DATASET, repo_type="dataset", revision=revision))
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "decontamination.json"):
        target = destination / name
        content = (source / name).read_bytes()
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
    manifest = json.loads((source / "archives.json").read_text())
    for item in manifest:
        archive = source / item["file"]
        marker = destination / ("." + archive.name + ".done")
        if marker.exists() and marker.read_text().strip() == item["sha256"]:
            continue
        digest = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive.name}")
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                rel = Path(member.name)
                if rel.is_absolute() or ".." in rel.parts or not member.isfile():
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(destination, filter="data")
        marker.write_text(item["sha256"] + "\n")


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
    p.add_argument("--pool", default=str(ROOT / "data" / "vision_opd_6k_decontaminated"))
    p.add_argument("--out", default=str(ROOT / "data" / "blinkopd_k500"))
    p.add_argument("--revision", default=DATASET_REVISION)
    p.add_argument("--download-only", action="store_true")
    a = p.parse_args()
    pool, out = Path(a.pool).resolve(), Path(a.out).resolve()
    if not (pool / "train.jsonl").exists():
        download_pool(pool, a.revision)
    rows = read_pool(pool / "train.jsonl")
    selected, training = validate(rows)
    for row in rows:
        image_path(pool, row["images"][0])
        image_path(pool, row["teacher_images"][0])
    if a.download_only:
        print(f"Verified {len(rows)} pool rows in {pool}")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    with (ROOT / "indices" / "first_glance.csv").open(newline="") as f:
        glances = {int(r["row"]): r for r in csv.DictReader(f)}
    if set(glances) != set(selected):
        raise ValueError("First-glance index differs from K500")
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
