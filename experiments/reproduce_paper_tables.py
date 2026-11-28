"""Reaggregate selected JAAMAS table rows from packaged arrays/action traces.

Run prepare_inputs.py first. No fitting, training or new latency measurement.
The command fails if a reproduced printed row differs from the editable paper.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/deadline_comm/runs"


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def normalize(text):
    return re.sub(r"\s+", "", text.replace("$", ""))


def percent(value):
    # Accuracy counts are exact rationals. Avoid binary float error at x.xx5;
    # the manuscript uses decimal half-up display rounding (e.g. 56.575 -> 56.58).
    return str(Decimal(str(round(float(value), 10))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def check_row(source, label, row):
    matches = [block for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.S)
               if "\\label{" + label + "}" in block]
    if len(matches) != 1 or normalize(row + r"\\") not in normalize(matches[0]):
        raise AssertionError(f"Paper row mismatch ({label}): {row}")


def external():
    path = ROOT / "experiments/deadline_comm/external_deadline/runs/mnist_deadline_20261123_002"
    by_deadline = defaultdict(list)
    for seed in range(3):
        with np.load(path / f"seed{seed}_episodes.npz", allow_pickle=False) as a:
            idx = np.arange(len(a["target"]))
            # Marginalize both stochastic policies, without reading saved scores.
            full = np.sum(a["speaker_probabilities"][:, :, None] * a["listener_by_message"], axis=1)[idx, a["target"]]
            zero = a["zero_probabilities"][idx, a["target"]]
            for deadline in (250, 500, 750, 1000, 1250):
                timely = a["ready_us"] + 220 <= deadline
                offered = a[f"random_offers_{deadline}us"]
                assert timely.sum() == offered.sum()
                informed = zero + timely * (full - zero)
                random = zero + (timely & offered) * (full - zero)
                by_deadline[deadline].append([17 * timely.mean(), 100 * informed.mean(),
                                               100 * random.mean(), 100 * (informed - random).mean()])
    rows = []
    for deadline, seed_values in sorted(by_deadline.items()):
        a = np.array(seed_values)
        mean, low, high = interval(a[:, 3])
        cost, full, random = a[:, :3].mean(axis=0)
        printed = f"{deadline:,} & {cost:.2f} & {full:.2f} & {random:.2f} & {mean:.2f} [{low:.2f}, {high:.2f}]"
        rows.append(dict(deadline_us=deadline, offered_B=cost, full_informed_pct=full,
                         random_pct=random, gain_pp=mean, ci_low=low, ci_high=high, printed=printed))
    return rows


def receiver():
    buckets = defaultdict(list)
    with (RUNS / "receiver_head_adaptation_20261123_001/episodes.jsonl").open() as stream:
        for line in stream:
            ep = json.loads(line)
            for step in ep["trace"]:
                accuracy = sum(x == step["bit"] for x in step["actions"][1:]) / 4
                buckets[ep["scheduler"], ep["arm"], ep["seed"]].append(
                    [accuracy, step["deadline_miss_including_diagnostic"]])
    rows = []
    for access, name in [("normal", "Normal RR"), ("source_aware", "Source-aware")]:
        for arm, label in [("original", "Unchanged"), ("adapted", "True-label adaptation"),
                           ("shuffled", "Shuffled-label adaptation")]:
            means = [np.mean(buckets[access, arm, seed], axis=0) for seed in range(10)]
            accuracy, miss = 100 * np.mean(means, axis=0)
            rows.append(dict(access=access, arm=arm, accuracy_pct=accuracy, miss_pct=miss,
                             printed=f"{name} & {label} & {percent(accuracy)} & {percent(miss)}"))
    return rows


def uncertain():
    rows = []
    for sd, cost in [(0, 0), (100, 100), (400, 400), (800, 400)]:
        offered, gains = [], []
        for seed in range(3):
            path = RUNS / f"mnist_uncertain_readiness_20261126_001/seed{seed}_cells.npz"
            with np.load(path, allow_pickle=False) as a:
                ready = a["ready_us"]
                if sd:
                    p = np.array([.5 * math.erfc(-(750 - 220 - cost - r) / (sd * math.sqrt(2))) for r in ready])
                else:
                    p = (ready + cost + 220 <= 750).astype(float)
                reward_delta = a["full_reward"] - a["zero_reward"]
                informed_delta = p * (ready + cost + 220 <= 750) * reward_delta
                random_delta = p.mean() * (ready + 220 <= 750) * reward_delta
                offered.append(17 * p.mean())
                gains.append(100 * (informed_delta - random_delta).mean())
        mean, low, high = interval(gains)
        wire = float(np.mean(offered))
        rows.append(dict(noise_sd_us=sd, decision_cost_us=cost, offered_B=wire,
                         gain_pp=mean, ci_low=low, ci_high=high,
                         printed=f"{sd} & {cost} & {wire:.3f} & {mean:.2f} [{low:.2f},{high:.2f}]"))
    return rows


def hard_fallback():
    # Accumulate each process/seed independently; processes are not extra seeds.
    buckets = defaultdict(lambda: np.zeros(3))
    for repeat in range(3):
        path = RUNS / f"broadcast_hard_deadline_20261126_001/repeat{repeat}/episodes.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            for line in stream:
                ep = json.loads(line)
                key = (ep["scheduler"], ep["fallback"], ep["post_cost_us"], ep["arm"], ep["seed"], repeat)
                for step in ep["trace"]:
                    accuracy = sum(x == step["bit"] for x in step["actions"][1:]) / 4
                    miss = step["cpu_total_ns"] + step["wait_ns"] + step["post_cost_ns"] > 2000000
                    buckets[key] += [accuracy, miss, 1]

    def stats(access, fallback, cost):
        repeats = range(3) if access == "normal" else range(1)
        arms = {}
        for arm in ["original", "adapted"]:
            seed_values = []
            for seed in range(10):
                processes = [buckets[access, fallback, cost, arm, seed, rep] for rep in repeats]
                assert all(v[2] == 400 for v in processes)
                seed_values.append(np.mean([v[:2] / v[2] for v in processes], axis=0))
            arms[arm] = 100 * np.array(seed_values)
        mean, low, high = interval(arms["adapted"][:, 0] - arms["original"][:, 0])
        return dict(access=access, fallback=fallback, post_cost_us=cost,
                    original_pct=float(arms["original"][:, 0].mean()),
                    adapted_pct=float(arms["adapted"][:, 0].mean()), gain_pp=mean,
                    ci_low=low, ci_high=high,
                    original_miss_pct=float(arms["original"][:, 1].mean()),
                    adapted_miss_pct=float(arms["adapted"][:, 1].mean()))

    rows = []
    for fallback, cost in [("ordinary", 0), ("hold", 0), ("zero", 0), ("hold", 250),
                           ("zero", 250), ("ordinary", 500), ("hold", 500), ("zero", 500)]:
        r = stats("normal", fallback, cost)
        if cost == 500 and fallback != "ordinary":
            assert r["gain_pp"] == r["ci_low"] == r["ci_high"] == 0
            r["printed"] = f"Hold / zero & 500 & {r['original_pct']:.2f} & {r['adapted_pct']:.2f} & 0 [0,0]"
        else:
            r["printed"] = (f"{fallback.title()} & {cost} & {percent(r['original_pct'])} & {percent(r['adapted_pct'])} & "
                            f"{percent(r['gain_pp'])} [{r['ci_low']:.2f},{r['ci_high']:.2f}]")
        rows.append(r)
    return rows, [stats("source_aware", fallback, 0) for fallback in ("ordinary", "hold", "zero")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/local_recompute/paper_tables")
    args = parser.parse_args()
    main_source = (ROOT / "results/table_sources/jaamas_main.tex").read_text()
    supp_source = (ROOT / "results/table_sources/jaamas_supplement.tex").read_text()
    hard, source_aware = hard_fallback()
    tables = {"external_deadlines": (main_source, external()),
              "receiver_controls_acting_rows": (main_source, receiver()),
              "uncertain_readiness": (supp_source, uncertain()),
              "hard_action": (supp_source, hard)}
    checked = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (source, rows) in tables.items():
        label = "tab:" + name.replace("_acting_rows", "")
        for row in rows:
            check_row(source, label, row["printed"])
            checked += 1
        with (args.output / f"{name}.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"PASS: {label}: {len(rows)} reaggregated rows match the manuscript.")
    primary = next(r for r in hard if r["fallback"] == "hold" and r["post_cost_us"] == 0)
    for key in ("original_miss_pct", "adapted_miss_pct"):
        assert f"{primary[key]:.3f}\\%" in main_source
    assert all(r["gain_pp"] == 0 and r["original_miss_pct"] == r["adapted_miss_pct"] == 100
               for r in source_aware if r["fallback"] != "ordinary")
    report = dict(passed=True, table_ids=["tab:" + n.replace("_acting_rows", "") for n in tables],
                  computed_rows=checked, distinct_printed_rows=checked - 1,
                  receiver_probe_rows_in_this_export=False, primary_hard_hold=primary,
                  source_aware=source_aware, training=False, new_latency_measurement=False,
                  manuscript_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [ROOT / "results/table_sources/jaamas_main.tex", ROOT / "results/table_sources/jaamas_supplement.tex"]})
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: primary miss rates and source-aware failure boundary. Output: {args.output}")


if __name__ == "__main__":
    main()
