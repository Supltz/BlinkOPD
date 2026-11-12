import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CUTS = ("0", "25", "50", "75", "100")
METHODS = ("vopd", "opdv", "vad", "cvopd", "aha", "blink")
LABELS = {"vopd": "Vision-OPD", "opdv": "OPD-V", "vad": "VAD", "cvopd": "CV-OPD", "aha": "OPD-Aha", "blink": "BlinkOPD"}
REFS = ("mismatch", "noise", "N", "black", "txt", "cut")
REF_LABELS = {"mismatch": "Mismatched Image", "noise": "Gaussian Noise", "N": "Gray Image", "black": "Black Image",
              "txt": "Text Only", "cut": "Eyes Closed (Ours)"}
AHA_BETA = 4.0
BLINK_BETA, BLINK_GAMMA = 3.4, 2.0
CV_MARGIN, CV_WEIGHT = 0.1, 0.1
TOPK = 100


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def top(d):
    return max(d, key=d.get)


def lg(d, opts):
    return np.log(np.clip(np.array([d.get(o, 0.0) for o in opts], float), 1e-12, None))


def lo(d, g, a):
    return np.log(max(d.get(g, 0), 1e-12)) - np.log(max(d.get(a, 0), 1e-12))


def jsd(a, b):
    import torch
    m = torch.logaddexp(a, b) - math.log(2.0)
    return 0.5 * ((a.exp() * (a - m)).sum(-1) + (b.exp() * (b - m)).sum(-1))


def effective_target(T, N, gate, w, iters=400, lr=0.05):
    import torch
    z = T.clone().double().requires_grad_(True)
    Td, Nd, g = T.double(), N.double(), gate.double()
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        p = torch.log_softmax(z, -1)
        jt = jsd(p, Td)
        loss = (jt + g * w * torch.relu(CV_MARGIN + jt - jsd(p, Nd))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.log_softmax(z, -1).float()


def opt_from_sup(lq, idx, letter_ids, opts):
    p = np.exp(np.asarray(lq[:TOPK], float))
    ix = np.asarray(idx)
    raw = {o: float(p[np.isin(ix, letter_ids[o])].sum()) for o in opts}
    s = sum(raw.values())
    return {o: v / s for o, v in raw.items()} if s > 0 else {o: 1 / len(opts) for o in opts}


def cvopd_effective(recs):
    import torch
    allj = np.concatenate([np.asarray(ro["tok_jsd_ST"], float) for r in recs for ro in r["rollouts"] if ro["tok_jsd_ST"]])
    thr = float(allj.mean())
    frac = float((allj > thr).mean())
    w = CV_WEIGHT / max(frac, 1e-9)
    keys, T, N, G = [], [], [], []
    for r in recs:
        for ro in r["rollouts"]:
            for cu in CUTS:
                cv = ro["probes"][cu]["cv"]
                keys.append((r["set"], r["row"], ro["k"], cu))
                T.append(cv["T"])
                N.append(cv["N"])
                G.append(float(cv["jsd_ST"] > thr))
    T, N, G = torch.tensor(T), torch.tensor(N), torch.tensor(G)
    Q = torch.cat([effective_target(T[i:i + 4096], N[i:i + 4096], G[i:i + 4096], w) for i in range(0, len(keys), 4096)]).numpy()
    by = {(r["set"], r["row"]): r for r in recs}
    eff = {}
    for key, q in zip(keys, Q):
        r = by[key[:2]]
        ro = next(x for x in r["rollouts"] if x["k"] == key[2])
        eff[key] = opt_from_sup(q, ro["probes"][key[3]]["cv"]["idx"], r["letter_ids"], r["options"])
    gate = {"threshold": thr, "gated_token_fraction": frac, "hinge_weight": w, "probe_gated_fraction": float(G.mean())}
    return eff, gate


def probe_numbers(recs):
    eff, gate = cvopd_effective(recs)
    fol = {m: {c: [] for c in CUTS} for m in METHODS}
    den = {m: {c: [] for c in CUTS} for m in METHODS}
    surv = {m: {c: [] for c in CUTS} for m in METHODS}
    ends = ("50", "75", "100")
    dT = {e: 0.0 for e in ends}
    dR = {e: {v: 0.0 for v in REFS} for e in ends}
    n_roll = n_correct = n_filtered = 0
    for r in recs:
        gt, opts = r["gt"], r["options"]
        g = opts.index(gt)
        chosen = [ro["chosen"] for ro in r["rollouts"] if ro["chosen"] in opts]
        filtered = bool(chosen) and all(c != gt for c in chosen)
        n_filtered += filtered
        for ro in r["rollouts"]:
            n_roll += 1
            ch = ro["chosen"]
            if ch not in opts:
                continue
            if ch == gt:
                n_correct += 1
                continue
            P = ro["probes"]
            a = opts.index(ch)
            b = BLINK_BETA * (BLINK_GAMMA if top(P["0"]["T"]) != ch else 1.0)
            in_k = top(P["0"]["T"]) == gt
            for cu in CUTS:
                lT, lN, lC = lg(P[cu]["T"], opts), lg(P[cu]["N"], opts), lg(P[cu]["cut"], opts)
                if filtered:
                    cv = P[cu]["cv"]
                    lE = lg(opt_from_sup(cv["S"], cv["idx"], r["letter_ids"], opts), opts)
                else:
                    lE = lg(eff[(r["set"], r["row"], ro["k"], cu)], opts)
                tg = {"vopd": lT, "vad": lg(P[cu]["vad"], opts), "cvopd": lE,
                      "opdv": lT if P[cu]["opdv"][ch] > 0 else lg(P[cu]["S"], opts),
                      "aha": (1 + AHA_BETA) * lT - AHA_BETA * lN, "blink": (1 + b) * lT - b * lC}
                for m in METHODS:
                    am = int(np.argmax(tg[m]))
                    fol[m][cu].append(am == a)
                    den[m][cu].append(am == g)
                    if in_k:
                        surv[m][cu].append(float(tg[m][g] - tg[m][a]))
            if in_k:
                for e in ends:
                    dT[e] += lo(P["0"]["T"], gt, ch) - lo(P[e]["T"], gt, ch)
                    for v in REFS:
                        dR[e][v] += lo(P["0"][v], gt, ch) - lo(P[e][v], gt, ch)
    n_wrong = len(fol["vopd"]["0"])
    n_k = len(surv["vopd"]["0"])
    return {
        "counts": {"questions": len(recs), "rollouts": n_roll, "student_wrong": n_wrong, "student_correct": n_correct,
                   "K_teacher_knew": n_k, "cvopd_filtered_questions": n_filtered},
        "cvopd_gate": gate,
        "fig2a_follow_rate_pct": {m: [100 * float(np.mean(fol[m][c])) for c in CUTS] for m in METHODS},
        "fig2b_survival_score": {m: [float(np.mean(surv[m][c])) for c in CUTS] for m in METHODS},
        "fig5_corrective_density_pct": {m: [100 * float(np.mean(den[m][c])) for c in CUTS] for m in METHODS},
        "fig3a_residual_drift_pct": {v: 100 * (1 - dR["100"][v] / dT["100"]) for v in REFS},
        "appC_residual_drift_pct": {e: {v: 100 * (1 - dR[e][v] / dT[e]) for v in REFS} for e in ends},
    }


def etoken_numbers(rows):
    keys = ("real",) + REFS
    return {"questions": len(rows), "fig3b_false_belief": {k: float(np.mean([r[k]["E"] for r in rows])) for k in keys},
            "letter_mass": {k: float(np.mean([r[k]["letter"] for r in rows])) for k in keys}}


def write_csv(path, out):
    rows = []
    for key, panel in (("fig2a_follow_rate_pct", "fig2a"), ("fig2b_survival_score", "fig2b"), ("fig5_corrective_density_pct", "fig5")):
        for m in METHODS:
            for c, v in zip(CUTS, out[key][m]):
                rows.append((panel, LABELS[m], c, v))
    for v in REFS:
        rows.append(("fig3a", REF_LABELS[v], "100", out["fig3a_residual_drift_pct"][v]))
    for e in ("50", "75"):
        for v in REFS:
            rows.append(("appC", REF_LABELS[v], e, out["appC_residual_drift_pct"][e][v]))
    if "etoken" in out:
        fb = out["etoken"]["fig3b_false_belief"]
        for v in REFS:
            rows.append(("fig3b", REF_LABELS[v], "", fb[v]))
        rows.append(("fig3b", "Real Image", "", fb["real"]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "series", "prefix_pct", "value"])
        for p, s, x, v in rows:
            w.writerow([p, s, x, f"{v:.6f}"])


def table(out):
    L = ["Fig. 2a follow rate (%) / Fig. 2b survival score / Fig. 5 corrective density (%), prefix read 0/25/50/75/100%"]
    for key, fmt in (("fig2a_follow_rate_pct", "{:.1f}"), ("fig2b_survival_score", "{:.2f}"), ("fig5_corrective_density_pct", "{:.1f}")):
        L.append(key)
        for m in METHODS:
            L.append(f"  {LABELS[m]:<11}" + " ".join(f"{fmt.format(v):>7}" for v in out[key][m]))
    L.append("Fig. 3a residual drift (%) at the whole prefix; Appendix C at 50% / 75%")
    for v in REFS:
        L.append(f"  {REF_LABELS[v]:<19} {out['fig3a_residual_drift_pct'][v]:5.1f}   50%: {out['appC_residual_drift_pct']['50'][v]:5.1f}"
                 f"   75%: {out['appC_residual_drift_pct']['75'][v]:5.1f}")
    if "etoken" in out:
        fb = out["etoken"]["fig3b_false_belief"]
        L.append("Fig. 3b false belief (probability of the 'not present' option E at the first answer token)")
        for v in ("real",) + REFS:
            L.append(f"  {('Real Image' if v == 'real' else REF_LABELS[v]):<19} {fb[v]:.3f}")
    c = out["counts"]
    L.append(f"counts: {c}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "probe"))
    ap.add_argument("--readouts", default="")
    ap.add_argument("--etoken", default="")
    a = ap.parse_args()
    out_dir = Path(a.out)
    readouts = Path(a.readouts) if a.readouts else out_dir / "readouts.jsonl.gz"
    etoken = Path(a.etoken) if a.etoken else out_dir / "etoken.jsonl"
    out = probe_numbers(read_jsonl(readouts))
    if etoken.is_file():
        out["etoken"] = etoken_numbers(read_jsonl(etoken))
    out["cuts_pct"] = [int(c) for c in CUTS]
    out["labels"] = {**LABELS, **REF_LABELS}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    write_csv(out_dir / "numbers.csv", out)
    print(table(out))
    print(f"wrote {out_dir / 'numbers.json'} and {out_dir / 'numbers.csv'}")


if __name__ == "__main__":
    main()
