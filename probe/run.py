import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "vision_opd_6k"))
    p.add_argument("--eval-data-dir", default=str(ROOT / "data" / "eval"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "probe"))
    p.add_argument("--rollouts", default="")
    p.add_argument("--skip-etoken", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    gpus = a.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(gpus) != len(set(gpus)):
        p.error("--gpus must contain distinct GPU indices")
    common = ["--out", a.out, "--model", a.model, "--data-dir", a.data_dir, "--eval-data-dir", a.eval_data_dir]
    if a.skip_etoken:
        common += ["--skip-etoken"]
    if a.rollouts:
        common += ["--rollouts", a.rollouts]
    processes = []
    if not a.plot_only:
        if not (Path(a.data_dir) / "train.jsonl").is_file():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
                            "--pool", a.data_dir, "--check-only"], check=True)
        if not a.skip_etoken and not (Path(a.eval_data_dir) / "MME_RealWorld_CN.json").is_file():
            subprocess.run([sys.executable, str(ROOT / "eval" / "prepare_data.py"),
                            "--data-dir", a.eval_data_dir, "--benchmarks", "mme-realworld-cn"], check=True)
        identity = {"model": a.model, "pool_sha256": hashlib.sha256((Path(a.data_dir) / "train.jsonl").read_bytes()).hexdigest(),
                    "retained_index_sha256": hashlib.sha256((ROOT / "indices" / "decontaminated_pool.csv").read_bytes()).hexdigest(),
                    "items_sha256": hashlib.sha256((ROOT / "probe" / "items.csv").read_bytes()).hexdigest(),
                    "benchmark_index_sha256": hashlib.sha256((ROOT / "probe" / "mmerw_cn_300.csv").read_bytes()).hexdigest(),
                    "template_sha256": hashlib.sha256((ROOT / "chat_templates" / "perception_chat_template_qwen35.jinja").read_bytes()).hexdigest(),
                    "readout_sha256": hashlib.sha256((ROOT / "probe" / "readout.py").read_bytes()).hexdigest()}
        destination = Path(a.out)
        destination.mkdir(parents=True, exist_ok=True)
        receipt = destination / "run_config.json"
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise ValueError("Probe model, data or readout code changed; choose a new --out directory")
        receipt.write_text(json.dumps(identity, indent=2) + "\n")
        try:
            for shard, gpu in enumerate(gpus):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                processes.append(subprocess.Popen([sys.executable, str(ROOT / "probe" / "readout.py"), "worker",
                    "--shard", str(shard), "--num-shards", str(len(gpus)), *common], env=env))
            codes = [process.wait() for process in processes]
            if any(codes):
                raise RuntimeError(f"Probe worker failed: {codes}. Rerun the same command to resume.")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        subprocess.run([sys.executable, str(ROOT / "probe" / "readout.py"), "merge", *common], check=True)
        subprocess.run([sys.executable, str(ROOT / "probe" / "stats.py"), "--out", a.out], check=True)
    subprocess.run([sys.executable, str(ROOT / "probe" / "plot.py"), "--out", a.out], check=True)


if __name__ == "__main__":
    main()
