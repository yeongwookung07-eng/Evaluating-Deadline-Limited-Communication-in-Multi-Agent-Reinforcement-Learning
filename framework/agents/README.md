# agents

Shared recurrent agent, episode buffer and TD learner dependencies; unchanged from the research implementation.

Inputs: repository-relative YAML settings, fixed checkpoints and matched reset seeds where required. Outputs: code-specific JSON/CSV estimates, traces or test assertions. Preserve checkpoint, cache, timing and replay controls when interpreting results. No new learning objective or universal scheduler advantage is claimed.

This is a selected source snapshot. See the root `REPRODUCE.md` for supported commands/dependencies and `EVIDENCE.md` for evidence, seeds and limitations. Existing study artifacts are retained separately from new runs; heavyweight inputs are omitted from this source snapshot and remain in the original research workspace.
