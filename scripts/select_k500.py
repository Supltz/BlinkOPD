import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_pool(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 6220:
        raise ValueError(f"Expected the 6,220-row decontaminated pool, found {len(rows)}")
    return rows


def select(rows, size=500, stages=(), seed=42, repeats=6, row_seed=44):
    indices = list(range(len(rows)))
    for count in stages:
        keys = sorted({Path(rows[i]["original_images"][0]).name for i in indices})
        chosen = set(random.Random(seed).sample(keys, count))
        indices = [i for i in indices if Path(rows[i]["original_images"][0]).name in chosen]
        if len(indices) != count:
            raise ValueError("Expected one question per original image")
    if not 0 < size <= len(indices) or repeats < 1:
        raise ValueError("Invalid sample size or repeat count")
    random.Random(seed).shuffle(indices)
    selected = indices[:size]
    training = selected * repeats
    random.Random(row_seed).shuffle(training)
    return selected, training


def records(rows, selected):
    return [{"selection_order": n, "row": i, "image": Path(rows[i]["images"][0]).name,
             "original_image": Path(rows[i]["original_images"][0]).name,
             "question_sha256": hashlib.sha256(rows[i]["extra_info"]["question"].encode()).hexdigest()}
            for n, i in enumerate(selected)]


def validate(rows, index=ROOT / "indices" / "k500.csv"):
    with Path(index).open(newline="") as f:
        actual = list(csv.DictReader(f))
    selected = [int(r["row"]) for r in actual]
    if len(selected) != 500 or len(set(selected)) != len(selected) or any(i < 0 or i >= len(rows) for i in selected):
        raise ValueError("Invalid K500 index")
    expected = records(rows, selected)
    if actual != [{k: str(v) for k, v in r.items()} for r in expected]:
        raise ValueError("K500 selection differs from the released index; check the pool version and row order")
    training = selected * 6
    random.Random(44).shuffle(training)
    return selected, training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=str(ROOT / "data" / "vision_opd_6k_decontaminated" / "train.jsonl"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "selection"))
    p.add_argument("--check", action="store_true")
    p.add_argument("--stages", type=int, nargs="*", default=[])
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--row-seed", type=int, default=44)
    p.add_argument("--repeats", type=int, default=6)
    a = p.parse_args()
    rows = read_pool(a.pool)
    selected, training = select(rows, a.size, a.stages, a.seed, a.repeats, a.row_seed)
    if a.check and (selected, training) != validate(rows):
        raise ValueError("Selection differs from the released K500 index; use the historical sampling arguments in README")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = records(rows, selected)
    with (out / "k500.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    with (out / "training_rows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["training_row", "pool_row"])
        w.writerows(enumerate(training))
    print(f"Selected {len(selected)} questions; produced {len(training)} rows")


if __name__ == "__main__":
    main()
