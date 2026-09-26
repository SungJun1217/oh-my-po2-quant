# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

The repository contains no code yet: no build system, dependencies, tests or lint config. When you add tooling, document the build/test commands here (including how to run a single test).

## What this project is

A research codebase for Post-Training Quantization under strict hardware constraints: INT8 weights and activations, symmetric, per-tensor, **power-of-two scales (2^k)**, configurable bias precision (INT8/16/32) and a separate accumulator width. "Po2" means the *scale* is a power of two. It does not mean log-quantization of the values.

## Where the rules live

- `AGENTS.md`: the full spec (40 sections). It is the source of truth when a rule is ambiguous.
- `.claude/rules/*.md`: condensed rules of 200 chars or fewer each, auto-loaded. They are split from AGENTS.md, so keep the two in sync when either changes.

## Intended architecture (from AGENTS.md)

Build around these seams when adding code:

1. **Hardware profile/config** (`configs/hw_profile.yaml`, provisional DPU-like defaults from `docs/hw/npu-op-survey.md`; `UNVERIFIED-HW` values await the vendor): a single explicit config (bits, signed range `[-127,127]` vs `[-128,127]`, rounding mode, saturation, bias bits + scale rule, accumulator bits, Po2-only requant, `image_divisor: 256`). Every quantizer reads behavior from it. Nothing is hard-coded.
2. **Preprocessing**: shared by calibration and deployment (default `uint8 * 2^-8`). Any folding of mean/std into the first layer must pass an FP32 equivalence test before quantization.
3. **Integer datapath simulator**: INT8×INT8 → wide accumulator → quantized bias add (`Sb = 2^(kx+kw)`) → Po2 requant shift → saturation. Use it in place of pure fake-quant.
4. **Exponent search**: discrete search over `k` candidates (configurable, default `base_k±4`). It goes up the ladder MinMax → tensor MSE → clipping-aware → layer reconstruction → joint (kx, kw) with the bias constraint. All candidate metrics are recorded, not only the winner.
5. **Evaluation**: task-level metrics with three baselines (FP32 `/255`, FP32 `/256`, INT8 Po2 `/256`), so that preprocessing loss, PTQ loss and bias loss are reported separately.

## Environments

- This Mac (M1 Max, no CUDA): code, unit tests, smoke tier on CPU.
- Separate CUDA GPU server: calibration, screening and gate runs (deterministic CUDA kernels). Not set up yet.
- Datasets (ImageNet, COCO 2017) live under `$PO2_DATA_ROOT`, never in the repo. Layout is in `benchmarks/phase1.yaml`.

## Current phase (Phase 1)

Conv-based networks from `benchmarks/phase1.yaml`. INT8 W/A, per-tensor Po2, **INT8 bias**, PTQ only (unlabeled calibration; exponent search, BN fold, CLE, bias correction, AdaRound/BRECQ-style rounding allowed). Goal: every dev model ≤ 1% relative loss vs FP32 /255, and ≥ 90% of held-out models. The method must be model-agnostic. Held-out results (`docs/results/**/heldout/`) must never feed back into design.

Evaluation tiers (`benchmarks/phase1.yaml` `eval_tiers`): **smoke** (100 samples, pipeline check only), **screening** (frozen 5k stratified ImageNet subset matched on FP32 accuracy and margin; full COCO/VOC val; adds FP32-agreement and logit KL) for in-cycle comparison, **gate** (full val, 5 calibration subsets), the only tier that can ACCEPT. Screening is trusted only after the baseline proxy check (Spearman ≥ 0.9 vs gate).

## Subagent workflow

Defined in `.claude/agents/`. Subagents can't call each other, so the main session (or the workflow) runs the pipeline and passes file paths between steps:

1. `po2-algorithm-designer` (opus) → `docs/design/<topic>.md`
2. `po2-architect` (opus) reviews the design → `docs/plans/<topic>.md`
3. `po2-implementer` (sonnet) implements one plan step at a time with tests
4. `po2-reviewer` (opus, read-only) reviews each step's diff; blockers go back to step 3 (or step 2 if the plan is wrong)
5. `po2-perf-evaluator` (opus) evaluates **one model** per call (modes: baseline / dev / heldout) with deterministic scripts in `experiments/` → `docs/results/<topic>/<split>/<model>.md`
6. `po2-failure-analyst` (opus) aggregates dev results into a failure taxonomy by structure → `docs/analysis/<topic>-c<cycle>.md`, the designer's input for the next cycle

Saved workflow `.claude/workflows/po2-research-cycle.js` automates this: subset build → baseline sweep → proxy check → design → plan → implement/review → dev screening → (all pass) dev gate → failure analysis → loop; when every dev model is ACCEPT at gate it runs the held-out gate once and stops. Args: `topic`, `goal`, optional `stopAfter` (baseline|design|plan|implement|dev|all), `maxCycles`=5, `maxFixRounds`=3, `evalConcurrency`=1 (raise only with separate GPUs), `baseline`=true.

## Branching

- `main`: releases only. Receives PRs from `develop`; tag each release (e.g. `v0.1-phase1`) and attach the gate result table to the release notes.
- `develop`: integration. Receives PRs from `feature/*` only; no direct commits.
- `feature/<topic>`: one workflow run / topic, branched from `develop`. Design, plan, code, eval scripts and results are all committed here.
- `main` and `develop` are protected (admins included): PR required, CI checks `branch-policy`, `repo-checks`, `python` must pass, review conversations must be resolved, no force-push. The repo allows merge commits only and deletes head branches after merge.
- Merge with **merge commits, not squash**: result reports reference commit SHAs, and squashing would orphan them.
- The workflow creates `feature/<topic>`, commits after each stage (each implementation step only after review passes), and pushes + opens the PR to `develop` only when run with `openPr: true`. Measured eval runs must happen on a clean, committed tree.

## GitHub automation

- `.github/workflows/ci.yml`: branch-flow check on PRs, rules ≤ 200 chars, YAML/workflow-script parsing, and ruff + pytest once `pyproject.toml` exists. Zoo/GPU evaluations do not run in CI.
- `.github/workflows/claude.yml` (`@claude` in comments, members only) and `claude-review.yml` (automatic PR review with inline comments). Both need the `CLAUDE_CODE_OAUTH_TOKEN` repo secret.
- `.github/dependabot.yml`: weekly updates targeting `develop`. torch/torchvision/numpy/scipy/onnx/ultralytics get patch updates only, because numerics changes require a baseline + gate re-run.
