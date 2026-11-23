# Reviewer release — 2026-11-23

This standalone folder is the current paper/evidence snapshot. No git checkout, original absolute path or GPU is needed for CPU verification. The current English paper is `experiments/deadline_comm/publication/kbs/kbs_main.pdf`; `RELEASE_MANIFEST.json` binds it to the included results and code. `evidence/revision_20261123/EVIDENCE_MAP.json` maps each new result to its entrypoint/config/artifacts. Historical failed runs and gates remain explicit.

Run at this folder's root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_submission.py --recompute --tests
```

This checks every release file, recomputes original probes/editorial statistics, runs bounded original frozen-policy replay, recomputes structural/history results from raw logs, audits actual receiver actions/returns/delivery, loads the six selected external speaker/listener models, and runs original plus new focused tests. It does not rerun policy training, hardware timing or all evaluation campaigns. The old `REVIEWER_GUIDE.md` documents original cohorts; its earlier artifact limits are supplemented here.

MNIST images/labels and third-party paper/supplement PDFs are intentionally not redistributed. To run external image-based evaluation separately, obtain the four gzip inputs:

```bash
python scripts/fetch_mnist.py
python scripts/fetch_mnist.py --check-only
python scripts/verify_submission.py --external-data experiments/deadline_comm/external_biases/data
```

The optional external-data check regenerates predictions and deadline comparisons from final models in a temporary output directory; it does not train. If the recorded external learning gate failed, it reports that blocked outcome without running inference conclusions. The downloader uses the mirror and checksums documented in the official PyTorch torchvision MNIST loader (`https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py`); the dataset's original location is `http://yann.lecun.com/exdb/mnist/`. External `data_hashes.json` records the exact training/evaluation file SHA256s. Final external checkpoints include their selected optimizer/RNG states. Intermediate optimizer/model snapshots and third-party downloads are omitted; milestone validation outputs and registration remain. Full training is bounded by its config and is not run by verification.

Six common 1,500-update checkpoints are also included so the parallel continuation requires no manual PID handoff. Preview the six commands with `python scripts/reproduce_external_continuation.py --output /tmp/mnist-reviewer-new`; add `--run` only to explicitly execute the registered, all-seed validation-gated continuation in that new directory. This helper uses the existing worker implementation and exact saved model/optimizer/RNG states. Later intermediate checkpoints are omitted. The original serial segment and manual handoff are retained as provenance, not the recommended reviewer execution path.

For any experiment rerun, copy its YAML and set a new `output`; existing run directories are evidence and entries refuse overwriting them. External deadline command details are in `experiments/deadline_comm/external_deadline/README.md`. Receiver-head fitting uses additional supervised labels, and normal/source-aware strata are distinct. Fixed-readiness results are not real-time deployment guarantees; source-aware receiver diagnostic costs cause deadline misses.

Build the paper using `python experiments/deadline_comm/publication/kbs/build.py` after installing TeX/CAS dependencies. This changes generated paper files, so perform checksum verification first or rebuild a separate copy. Model/data raw inputs are preserved exactly; wall-clock timings and newly generated PDF bytes need not reproduce across machines. Author declarations remain the authors' responsibility; no submission/upload is asserted.
