"""Monitor the day-two run and summarize byte counts from finalized conditions."""
import argparse
import ast
import csv
import json
import time
from collections import defaultdict
from pathlib import Path


def status(run):
    manifest = json.loads((run / "manifest.json").read_text())
    print(manifest["status"], manifest["phase"], "seed", manifest["seed"],
          manifest.get("condition", "") if manifest["phase"] == "evaluation" else "")
    for seed in manifest["config"]["seeds"]:
        log = run / f"seed{seed}/training_stdout.log"
        latest = None
        if log.exists():
            for line in reversed(log.read_text().splitlines()):
                if line.startswith("{") and line.endswith("}") and "'mean_return'" in line:
                    latest = ast.literal_eval(line)
                    break
        if latest:
            print(f"seed {seed}  update {latest['update']}  mean Return {latest['mean_return']:.2f}"
                  f"  successes {latest['successes']}  evaluation conditions"
                  f" {len(list(log.parent.glob('*/summary.json'))) - int((log.parent/'train/summary.json').exists())}")
        else:
            print(f"seed {seed}  {'waiting for first evaluation' if log.exists() else 'pending'}")
    return manifest["status"]


def analyze(run):
    rows = []
    for file in sorted(run.glob("seed*/*/steps.csv")):
        summary = file.with_name("summary.json")
        if not summary.exists():
            continue
        totals = defaultdict(lambda: [0, 0, 0, 0])
        with file.open() as handle:
            for step in csv.DictReader(handle):
                acc = totals[step["method"]]
                for i, value in enumerate((1, step["attempted_bytes"], step["delivered_bytes"], step["K_delivered"])):
                    acc[i] += int(value)
        for result in json.loads(summary.read_text()):
            n, offered, accepted, packets = totals[result["method"]]
            assert n == result["steps"] and accepted == 48 * packets
            rows.append(dict(seed=result["training_seed"], category=result["category"], condition=result["condition"],
                             method=result["method"], mean_return=result["mean_return"], steps=n,
                             offered_wire_B_step=offered/n, deadline_accepted_wire_B_step=accepted/n,
                             effective_payload_goodput_B_step=32*packets/n,
                             wasted_bandwidth_pct=100*(offered-accepted)/offered if offered else None))
    if rows:
        out = run / "goodput_by_seed.csv"
        tmp = out.with_suffix(".tmp")
        with tmp.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    while True:
        state = status(args.run)
        if state == "completed":
            analyze(args.run)
        if state != "running" or not args.follow:
            break
        time.sleep(60)
