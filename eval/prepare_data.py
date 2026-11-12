import argparse
import base64
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

BENCHMARK_FILES = {
    "vstar": "vstar.json",
    "zoombench": "zoombench.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
}

SOURCES = {
    "vstar": ("lmms-lab-encoder/vstar-bench", "b44023b4dca749ed8a76b85eb576627d05a1c174",
              ["data/test-00000-of-00001.parquet"]),
    "zoombench": ("inclusionAI/ZoomBench", "b788097e57d30510c6877824833234a73bf80d25", ["data/test.parquet"]),
    "hrbench-4k": ("DreamMr/HR-Bench", "83b9013d6293b85dc507e87199ca52517536939c", ["hr_bench_4k.parquet"]),
    "hrbench-8k": ("DreamMr/HR-Bench", "83b9013d6293b85dc507e87199ca52517536939c", ["hr_bench_8k.parquet"]),
    "mme-realworld": ("yifanzhang114/MME-RealWorld-Lmms-eval", "f157390129e8d2b60757eb68d340aade9446b52b",
                      ["data/*.parquet"]),
    "mme-realworld-cn": ("yifanzhang114/MME-RealWorld-CN-Lmms-eval", "37ac6f4e57ef211b37d1ee04a8f1268f1411b3e0",
                         ["data/*.parquet"]),
}

IMAGE_DIRS = {
    "vstar": "VStar_images",
    "zoombench": "ZoomBench_images",
    "hrbench-4k": "HRBench_4k_images",
    "hrbench-8k": "HRBench_8k_images",
    "mme-realworld": "MME_RealWorld_Full_images",
    "mme-realworld-cn": "MME_RealWorld_CN_images",
}


def download(benchmark, data_dir):
    repo, revision, patterns = SOURCES[benchmark]
    local = data_dir / "raw" / repo.replace("/", "_")
    snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=patterns, local_dir=str(local))
    return local


def write_image(data_dir, rel, data, overwrite=True):
    path = data_dir / rel
    if path.exists() and not overwrite:
        return
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(f"invalid image bytes for {rel}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def image_bytes(obj):
    return obj.get("bytes") if isinstance(obj, dict) else None


def build_vstar(local, data_dir):
    import pyarrow.parquet as pq

    table = pq.read_table(str(local / "data" / "test-00000-of-00001.parquet"),
                          columns=["image", "text", "label", "question_id", "category"])
    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    data = []
    for i, row in enumerate(table.to_pylist()):
        qid = row.get("question_id")
        if qid is None or str(qid).strip() == "":
            qid = i
        rel = f"{IMAGE_DIRS['vstar']}/{qid}.jpg"
        write_image(data_dir, rel, image_bytes(row.get("image") or {}))
        query = (row.get("text") or "").strip()
        if not query.endswith(post_prompt.strip()):
            query += post_prompt
        data.append({
            "images": [rel],
            "query": query,
            "response": (row.get("label") or "").strip().upper(),
            "question_id": qid,
            "category": row.get("category") or "unknown",
        })
    return data


def build_zoombench(local, data_dir):
    import pyarrow.parquet as pq

    table = pq.read_table(str(local / "data" / "test.parquet"), columns=["id", "query", "response", "image"])
    data = []
    for i, row in enumerate(table.to_pylist()):
        sid = str(row.get("id") or i)
        rel = f"{IMAGE_DIRS['zoombench']}/{sid}.jpg"
        write_image(data_dir, rel, image_bytes(row.get("image") or {}))
        data.append({
            "images": [rel],
            "query": (row.get("query") or "").strip(),
            "response": (row.get("response") or "").strip(),
        })
    return data


def build_hrbench(local, data_dir, benchmark):
    import pyarrow.parquet as pq

    name = "hr_bench_4k.parquet" if benchmark == "hrbench-4k" else "hr_bench_8k.parquet"
    table = pq.read_table(str(local / name),
                          columns=["index", "question", "answer", "A", "B", "C", "D", "category", "image"])
    data = []
    for i, row in enumerate(table.to_pylist()):
        idx = int(row.get("index", i))
        rel = f"{IMAGE_DIRS[benchmark]}/{idx:05d}.jpg"
        write_image(data_dir, rel, base64.b64decode(row.get("image") or ""))
        question = (row.get("question") or "").strip()
        options = []
        for letter in ["A", "B", "C", "D"]:
            option = (row.get(letter) or "").strip()
            if option:
                options.append(f"({letter}) {option}")
        query = question + " Select from the following choices.\n" + "\n".join(options) if options else question
        data.append({
            "index": idx,
            "question_id": idx,
            "images": [rel],
            "query": query,
            "response": (row.get("answer") or "").strip(),
            "category": (row.get("category") or "").strip(),
        })
    return data


def mme_realworld_query(question, option_lines, lang):
    if not option_lines:
        return question
    if lang == "cn":
        return (
            question
            + " 选项如下所示:\n"
            + "\n".join(option_lines)
            + "\n根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n"
            + "最佳答案为： "
        )
    return (
        question
        + " The choices are listed below:\n"
        + "\n".join(option_lines)
        + "\nSelect the best answer to the above multiple-choice question based on the image. "
        + "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
        + "The best answer is: "
    )


def build_mme_realworld(local, data_dir, benchmark):
    import pyarrow.dataset as ds

    lang = "cn" if benchmark == "mme-realworld-cn" else "en"
    dataset = ds.dataset(str(local / "data"), format="parquet")
    columns = ["bytes", "index", "question", "multi-choice options", "answer"]
    data = []
    with tqdm(total=dataset.count_rows(), desc=benchmark, unit="img") as bar:
        for batch in dataset.to_batches(columns=columns):
            for row in batch.to_pylist():
                idx = int(row.get("index", 0))
                rel = f"{IMAGE_DIRS[benchmark]}/{idx:06d}.jpg"
                write_image(data_dir, rel, base64.b64decode(row.get("bytes") or ""), overwrite=False)
                question = (row.get("question") or "").strip()
                options = row.get("multi-choice options") or []
                option_lines = [str(x).strip() for x in options if str(x).strip()]
                data.append({
                    "index": idx,
                    "question_id": idx,
                    "images": [rel],
                    "query": mme_realworld_query(question, option_lines, lang),
                    "response": (row.get("answer") or "").strip(),
                })
                bar.update(1)
    return data


def build(benchmark, local, data_dir):
    if benchmark == "vstar":
        return build_vstar(local, data_dir)
    if benchmark == "zoombench":
        return build_zoombench(local, data_dir)
    if benchmark in ("hrbench-4k", "hrbench-8k"):
        return build_hrbench(local, data_dir, benchmark)
    return build_mme_realworld(local, data_dir, benchmark)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_FILES))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        if benchmark not in BENCHMARK_FILES:
            raise SystemExit(f"unknown benchmark {benchmark}; choose from {', '.join(BENCHMARK_FILES)}")
        out_json = data_dir / BENCHMARK_FILES[benchmark]
        if out_json.exists():
            print(f"[data] {benchmark}: {out_json} exists")
            continue
        repo, revision, _ = SOURCES[benchmark]
        print(f"[data] {benchmark}: downloading {repo}@{revision[:12]}", flush=True)
        data = build(benchmark, download(benchmark, data_dir), data_dir)
        tmp = out_json.with_name(out_json.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_json)
        print(f"[data] {benchmark}: {len(data)} records -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
