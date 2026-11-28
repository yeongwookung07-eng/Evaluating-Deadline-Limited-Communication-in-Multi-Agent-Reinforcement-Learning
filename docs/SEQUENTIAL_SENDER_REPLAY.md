# Reproduce the two sequential sender interventions

Run these commands from the extracted review repository root on Linux. Both evaluations restore three fixed learned checkpoints. Each compares one sender substitution with the natural choice at a shared decision. The saved-data audit runs with Python's standard library. Fresh trajectories additionally require separate game runtimes.

## Check the archived trajectories

```bash
python3 experiments/prepare_inputs.py
audit=experiments/deadline_comm/publication/jaamas/audit_sender_transfers.py
runs=experiments/deadline_comm/runs
fresh=$(mktemp -d "${TMPDIR:-/tmp}/jaamas-sequential.XXXXXX")
python3 -B "$audit" --runs "$runs" --output "$fresh/archived-audit.json"
```

Preparation checks the file length and SHA-256 for each result and input, then restores the recorded run-relative paths. The audit checks episode, decision, prefix, sham and no-op evidence against raw trajectories and recomputes the registered estimands. It accepts `--arm predator` or `--arm terran` for a single environment. The commands below write new trajectories under `$fresh`. Keep the run directory names shown so the same audit can find them.

## Native SchedNet Predator–Prey

The upstream actor and scheduler use the TensorFlow 1 interface. The recorded CPU runtime used Python 3.10.20, TensorFlow CPU 2.15.1, NumPy 1.23.5, SciPy 1.15.3 and Gym 0.10.11. Set up the separate environment and obtain the authors' source at its recorded revision. The loader below copies only the files listed in the bundled source manifest and checks each byte digest before use.

```bash
native=experiments/deadline_comm/reviewer_validation
python3.10 -m venv "$fresh/predator-env"
predpy="$fresh/predator-env/bin/python"
"$predpy" -m pip install -r "$native/native_requirements_locked.txt"
git clone https://github.com/rhoowd/sched_net.git "$fresh/sched_net"
git -C "$fresh/sched_net" checkout --detach ffa03007cc654000a859856401231a986a01fbd0
python3 - "$fresh/sched_net" "$native/native_schednet" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
manifest = json.loads((target / 'SOURCE_MANIFEST.json').read_text())
for row in manifest['files']:
    src, dest = source / row['path'], target / row['path']
    if hashlib.sha256(src.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError(f'Upstream source mismatch: {src}')
    if dest.exists() and dest.read_bytes() != src.read_bytes():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
PY
"$predpy" experiments/deadline_comm/native_predator_sender/matched_sender_replay.py \
  --config experiments/deadline_comm/configs/native_predator_sender_replay.yaml \
  --output "$fresh/native_predator_sender_20261128_001"
python3 -B "$audit" --arm predator --runs "$fresh" --output "$fresh/predator-audit.json"
```

The run uses 50,000-step native checkpoints under `review_native_20261119_001/seed{0,1,2}_native/results/nn`. Native greedy scheduling and stochastic predator/prey actions are replayed on 24 common resets, with one of four senders at action 0 and, when reached, action 20. Every surviving episode is capped at 50 executed actions. Additional trajectories run uniform predator actions, zero scheduled payload, and the original scheduled payload passed through the same graph intervention. Episode, branch and compressed per-step records are written alongside the summary. The study took approximately 162 seconds on the recorded host.

## SMACv2 Terran

Use the pinned SMACv2 source, Blizzard's headless Linux StarCraft II 4.10 build 75689 and `10gen_terran.SC2Map`. The registered runtime provenance identifies the download URLs and SHA-256 for all three archives. The engine archive is about 4.1 GB. Read and accept the upstream StarCraft II Linux package terms before extracting the engine. The commands use the public archive password from the upstream setup. [SC2 Linux packages and terms](https://github.com/Blizzard/s2client-proto#linux-packages)

```bash
runtime="$fresh/sc2"
mkdir "$runtime"
python3 - "$runs/smacv2_runtime_20261119_001/provenance.json" "$runtime" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
for key, name in [('source_url', 'smacv2.zip'), ('maps_url', 'SMAC_Maps.zip'),
                  ('sc2_url', 'SC2.4.10.zip')]:
    archive = root / name
    subprocess.run(['curl', '-fL', '--retry', '2', '-o', str(archive),
                    record['config'][key]], check=True)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256'][name]:
        raise ValueError(archive)
PY
unzip -q "$runtime/smacv2.zip" -d "$runtime"
unzip -q -P iagreetotheeula "$runtime/SC2.4.10.zip" -d "$runtime"
export SC2PATH="$runtime/StarCraftII"
chmod +x "$SC2PATH/Versions/Base75689/SC2_x64"
mkdir -p "$SC2PATH/Maps/SMAC_Maps"
unzip -jq "$runtime/SMAC_Maps.zip" 10gen_terran.SC2Map -d "$SC2PATH/Maps/SMAC_Maps"
python3.10 -m venv "$fresh/terran-env"
terpy="$fresh/terran-env/bin/python"
"$terpy" -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
"$terpy" -m pip install numpy==1.26.4 scipy==1.15.3 protobuf==3.20.3 PyYAML==6.0.3 \
  pysc2==4.0.0 s2clientprotocol==5.0.17.98310.0 \
  "$runtime/smacv2-577ab5a2cff2391f8df582da5731ea9cd6adf3c6"
export CUDA_VISIBLE_DEVICES= PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$terpy" - "$fresh" <<'PY'
from pathlib import Path
import sys, yaml
root = Path(sys.argv[1])
src = Path('experiments/deadline_comm/configs/terran_sender_branch_replay.yaml')
cfg = yaml.safe_load(src.read_text())
out = root / 'terran_sender_branch_replay_20261128_001'
out.mkdir()
cfg['output'] = str(out)
(root / 'terran.yaml').write_text(yaml.safe_dump(cfg))
PY
module=experiments.deadline_comm.smac_protocol.terran_sender_branch_replay
"$terpy" -m "$module" --config "$fresh/terran.yaml" --phase diagnosis
"$terpy" -m "$module" --config "$fresh/terran.yaml" --phase branches
python3 -B "$audit" --arm terran --runs "$fresh" --output "$fresh/terran-audit.json"
```

The recorded execution ran PyTorch on CPU with Python 3.10.20 and the 2.5.1+cu121 wheel. Each 300,000-step checkpoint uses ten resets. The branch follows eight round-robin actions and compares five single-sender alternatives under the same configured deadline. The diagnosis records the historical reproduction failure and tests QuickSave/QuickLoad. Branch evaluation can reconstruct the complete prefix in serial fresh engine processes. Check the generated `summary.json` for the exact-replay gate before interpreting any suffix-return contrast. With both fresh runs available, execute the unfiltered audit on `$fresh` to check the two experiments together.
