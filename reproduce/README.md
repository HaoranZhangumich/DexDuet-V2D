# Reproduce the V2D results (G1 + Dex3)

These scripts reproduce the V2D results of the [DexDuet paper](https://arxiv.org/abs/2611.30495) for DexDuet, DexDuet without interaction rewards and CHORD (PPO) on
the 30 Tier-1 episodes of the V2D Challenge, Track 2, at the same training budget.

There are two routes:

1. **From our checkpoints** (fast): download the trained policies, then evaluate, score Contact-AUC and summarize.
   About 10 minutes per method and episode: 8 evaluator runs of under a minute each on one GPU, then a few minutes
   of CPU for the Contact-AUC score.
2. **Train from scratch**: prepare the data, train, then the same three steps. About 10 to 11 hours per run, with two or three runs sharing a GPU.

## Setup

You need:

- **NVIDIA video_to_data at commit `8c778f9`** with its LFS files
  (`git clone https://github.com/nvidia-isaac/video_to_data && git checkout 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef &&
  git lfs pull`). CHORD trains there, and every method is evaluated there.
- **Its FlashCHORD environment** (`cd robotic_grounding/flash_chord && uv sync --frozen --extra train`). Its Python is
  `PY`. This repository runs in the same environment.
- **Its robotic_grounding environment**, set up as robotic_grounding's README describes (NVIDIA's retargeter and the
  SOMA-X body model). Its Python is `PY_V2D`. It is needed only to prepare data and for the human side of Contact-AUC.
  `prepare_data.sh` runs the retargeter from `$FC_OFFICIAL/..`; set `V2D_RG` if your robotic_grounding folder is
  elsewhere.
- **The V2D submission kit**, extracted (`V2D_KIT`), and **the challenge data**, folder `track_2` (`V2D_TRACK2`),
  both from the V2D Challenge's own release.
- One NVIDIA GPU with about 16 GB free per training run (we ran two or three runs per GPU) and about 2 GB per evaluation.

Set these variables (every script checks the ones it needs; `reproduce/config.sh` lists them all):

```bash
export FC_OFFICIAL=/path/to/video_to_data/robotic_grounding/flash_chord
export PY=$FC_OFFICIAL/.venv/bin/python            # FlashCHORD environment
export PY_V2D=/path/to/robotic_grounding/env/bin/python
export V2D_KIT=/path/to/v2d_submission_kit
export V2D_TRACK2=/path/to/video_to_data_challenge/track_2
export V2D_DATA=/path/to/human_motion_data         # prepared Tier-1 inputs (prepare_data.sh writes them)
export RUNS=/path/to/runs                          # training outputs; results go to $RUNS/results
bash reproduce/setup_assets.sh                     # links the robot meshes and the SONIC policy into this repo
```

The Tier-1 inputs are needed by both routes, because the evaluator replays each episode's reference:

```bash
bash reproduce/prepare_data.sh --gpus 0,1,2,3      # all 30 episodes; about 4 minutes per episode and GPU
```

## Route 1: from our checkpoints

```bash
bash reproduce/download_checkpoints.sh              # -> checkpoints/v2d_g1_dex3/<method>/epNN/
bash reproduce/run_all.sh --from-release checkpoints --gpus 0,1,2,3
```

`download_checkpoints.sh` fetches `dexduet_v2d_g1_dex3.zip` from the Hugging Face repository
[oldTOM/dexduet-v2d-checkpoints](https://huggingface.co/oldTOM/dexduet-v2d-checkpoints) and checks every file against
`CHECKPOINTS.sha256`. The package holds all three methods. The zip must be the release that `CHECKPOINTS.sha256`
lists; if the Hugging Face repository holds another release, update this checkout, or set `DEXDUET_CKPT_REVISION`
to the repository revision (branch, tag or commit) of the release it lists.

## Route 2: train from scratch

```bash
bash reproduce/run_all.sh --gpus 0,1,2,3 --per-gpu 2
```

CHORD (PPO) on episode 07 trains with a non-finite guard: set `FC_GUARD` as Notes describe, and `run_all.sh` passes
`--nonfinite-guard` for that run (without `FC_GUARD` it refuses that one job, runs the others and ends with an error).
Or step by step, for one method and one episode:

```bash
bash reproduce/train.sh dexduet 5 0          # method, episode (0-29), GPU
bash reproduce/evaluate.sh dexduet 5 0
bash reproduce/contact_auc.sh dexduet 5 0
python reproduce/summarize.py                # after all episodes: $RUNS/results/summary.md
```

## Scripts

| Script | What it does |
|---|---|
| `setup_assets.sh` | Links (or copies, `--copy`) the robot meshes and the SONIC whole-body policy from `FC_OFFICIAL` (git holds neither). |
| `prepare_data.sh` | Converts the challenge's Tier-1 episodes into per-episode inputs (`tools/track2_to_soma_inputs.py`), runs NVIDIA's retargeter unchanged (`soma_to_g1.py --save --fps 30`) to make the G1 reference motion, and builds DexDuet's human targets for the two interaction rewards (`scripts/dexduet/`). |
| `train.sh <method> <ep> [gpu]` | Trains one run at the equal budget. It composes the full config first and refuses to train unless budget, seed, robot and reward weights are the paper's. `--check` only checks. Re-running resumes an unfinished run. `--nonfinite-guard`: CHORD (PPO) on episode 07 (see Notes). |
| `evaluate.sh <method> <ep> [gpu]` | Four runs of the official evaluator. Each policy is first exported to the official policy format. A run that stops with an error is started once more (the log records both tries). `--from-release DIR` evaluates our checkpoint. |
| `contact_auc.sh <method> <ep> [gpu]` | Four more evaluator runs with the simulated body poses recorded, the human hand side of the episode, and the Contact-AUC score. |
| `summarize.py` | Method means ± their spread over the evaluator runs, paired differences against DexDuet (bootstrap CI, win/tie/loss, sign test) and a per-episode table. |
| `run_all.sh` | All of the above for every method and episode, spread over GPUs. |
| `download_checkpoints.sh` | Downloads and checks our checkpoints. |

Methods: `dexduet`, `dexduet_no_interaction_rewards`, `chord_ppo`.

## What each method runs

All three use the same 30 episodes, seed 42, and 1,966,080,000 environment steps per episode
(20,000 iterations x 4,096 worlds x 24 steps).

| Method | Code | Command (besides data path, seed, output and logging) |
|---|---|---|
| DexDuet | this repository | `scripts/train_rl.py experiment=g1_recon_body_coord_rsi_r2_dexduet training.max_iterations=20000 objective.dexduet_targets_dir=$TARGETS` |
| DexDuet without interaction rewards | this repository | `scripts/train_rl.py experiment=g1_recon_body_coord_rsi_r2 training.max_iterations=20000` |
| CHORD (PPO) | official, unmodified; episode 07 with the non-finite guard (Notes) | `scripts/train_rl.py experiment=g1_recon_body_ppo` (its default budget is the one above) |

The DexDuet runs stop earlier than 20,000 iterations on some episodes (10 of 30 in the paper): a fixed 5,860
iterations after A-RSI has withdrawn all assistance. `final.json` names the checkpoint to evaluate. Episodes whose
support file holds no shape train with `scene.support=false` (column `support` of `episodes.tsv`), as the trainers'
own rule says.

## Notes

- **Not bit-reproducible.** GPU training and the official evaluator are not deterministic. That is why every value is
  the mean of four evaluator runs. Over the four runs, DexDuet's mean ADD-AUC ranged from 76.1 to 78.2; the ± of
  `summarize.py` and of [docs/DETAILS.md](../docs/DETAILS.md#results) is the standard deviation of these four run means.
  A newly trained run differs from ours by more than that on single episodes; the method means are what to compare.
- **CHORD (PPO), episode 07.** CHORD (PPO) uses NVIDIA's unmodified code on the other 29 episodes. On episode 07 it is
  trained with NVIDIA's code plus a guard that resets a simulated world whose state becomes non-finite
  (`patches/chord_ppo_nonfinite_guard.diff`; 14 resets in 20,000 iterations in our run): on this episode the physics
  state of a world occasionally becomes non-finite, and NVIDIA's trainer has no check for it. The patch is the guard
  as it ran; code that does not affect training is left out. It prints how many worlds were reset. To train this run, apply the patch to a separate
  copy of the official checkout, so that `FC_OFFICIAL` stays unmodified, and pass `--nonfinite-guard` (`run_all.sh`
  passes it when `FC_GUARD` is set):

  ```bash
  cp -a /path/to/video_to_data /path/to/video_to_data_guard
  git -C /path/to/video_to_data_guard/robotic_grounding/flash_chord apply "$PWD/reproduce/patches/chord_ppo_nonfinite_guard.diff"
  export FC_GUARD=/path/to/video_to_data_guard/robotic_grounding/flash_chord
  bash reproduce/train.sh chord_ppo 7 0 --nonfinite-guard
  ```

  `train.sh` refuses an `FC_GUARD` that is not at `8c778f9`, whose Python or YAML files differ from that commit other
  than by the patch (a changed file, or an added one under `src/` or `scripts/`), or whose other files listed in
  `tools/official_pins.sha256` differ. A virtual environment, `outputs/` or `wandb/` folder in the copy is fine. It also refuses to resume a run with the other code, and the normal path still refuses a patched
  `FC_OFFICIAL`. Evaluation and Contact-AUC use the unmodified `FC_OFFICIAL`, as for every other policy.
- **Episode 29** has no frame in which the human holds the object, so it has no Contact-AUC.
- **Hull cache.** FlashCHORD splits each object into convex pieces and caches them. All our training runs used one
  fixed cache (29 files, `HULL_CACHE.sha256`). Without it, FlashCHORD builds the pieces on first use; they can differ
  slightly. Set `HULL_CACHE` to a folder holding `convex/*.npz` to use a given cache.
- **Data paths inside checkpoints.** The evaluator reads the reference from the parquet path stored in the checkpoint.
  When that path does not exist on your machine, `evaluate.sh` evaluates a copy whose stored path points at your
  `V2D_DATA` (only the metadata changes; `ckpt.py relocate`). The result files record both.
- **Contact-AUC software.** The scorer was written for NumPy 1.x; it also runs with NumPy 2 (`trapz` is mapped to
  `trapezoid`). On the paper's recordings we checked, NumPy 1.26.4 / SciPy 1.15.3 and NumPy 2.4.6 / SciPy 1.17.1
  give identical scores. Set `PY_CAUC` to use another Python for scoring.
- **Checks we ran.** `summarize.py` on the paper's per-run result files reproduces every number of the V2D results
  table in [docs/DETAILS.md](../docs/DETAILS.md#results), including the ±, and its paired differences. The scorer on
  the paper's recordings reproduces the stored per-episode Contact-AUC exactly.
