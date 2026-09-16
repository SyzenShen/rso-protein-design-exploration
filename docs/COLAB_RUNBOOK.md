# COLAB_RUNBOOK — Running GPU compute on Google Colab

This project runs RSO + ProteinMPNN + AlphaFold validation **on Google Colab with an NVIDIA GPU runtime**. Local runs are limited to scaffolding, tests, collection, and plotting (see `RUNBOOK.md`).

## Step 1 — Open the notebook

On your local machine, make sure you have pulled latest of the repo and know the GitHub URL of your fork. Then go to [Google Colab](https://colab.research.google.com/). Choose:

**File → Upload notebook →** select `notebooks/01_rso_reproduction.ipynb` from your local clone.

Or, to open directly from GitHub with Colab's GitHub browser, paste the raw URL of your notebook (e.g. `https://github.com/YOUR_USER/rso-protein-design-exploration/blob/main/notebooks/01_rso_reproduction.ipynb`) into the GitHub tab.

## Step 2 — Sign in with Google

Colab requires a Google account. Free Colab can be enough for 100-aa reproduction (A100s are occasional). Paid tiers (Colab Pro / Pro+ / Pay-as-you-go) make 40-GB A100s reliably available.

## Step 3 — Select GPU runtime

Top menu: **Runtime → Change runtime type.** Under *Hardware accelerator* choose **GPU**. Under *Runtime shape* choose **High-RAM** if available. Click *Save*.

## Step 4 — Run the first cell and check the actual GPU

Run cell 01 "setup: check GPU + install deps". The cell prints `nvidia-smi` output. Expected something like:

```
GPU DETECTED: NVIDIA A100-SXM4-40GB, 40960 MiB
```

**If you see `No nvidia-smi` or `FATAL: No NVIDIA GPU`**: STOP. Go back to Step 3 — sometimes you need to terminate the session and reconnect to actually get the GPU. Do not proceed with CPU; it will take many hours and eventually OOM on AF2.

## Step 5 — (Optional) Mount Google Drive for persistence

If you are worried about Colab session disconnects mid-run, mount Drive so the results/ directory survives disconnections:

```python
from google.colab import drive
drive.mount('/content/drive')
import os; os.makedirs('/content/drive/MyDrive/rso_protein_results', exist_ok=True)
# and symlink/copy results there after each stage completes in the notebook.
```

Mounting is NOT required to run a short reproduction notebook; it is purely a safeguard. Do **not** commit Drive credentials anywhere.

## Step 6 — Run setup cell (ColabDesign install + repo clone)

The setup cell runs:
- `git clone https://github.com/sokrypton/ColabDesign.git`
- `pip install colabdesign ... biopython pyyaml pandas matplotlib seaborn`
- downloads the official AlphaFold params `alphafold_params_2022-12-06.tar` (~3.6 GB) via `aria2c` and extracts to `/content/params`
- compiles the Zhang-group `TMscore` binary into `/content/TMscore`

The pip step usually takes 2–5 minutes; the params download adds 3–10 minutes depending on the Colab network. Expected final lines: `AF2 params: ['params_model_1_ptm.npz', ..., 'params_model_5_ptm.npz', ...]` and `TMscore binary present: True`. Expect several warnings; actual errors with tracebacks should be investigated. If the params step is skipped or interrupted, the first RSO cell fails with `AssertionError: ERROR: no model params defined` — re-run the setup/params cell.

If you want to use YOUR fork's configs/notebooks rather than the shipped notebook-internal defaults, also clone your fork:

```python
!git clone YOUR_FORK_URL /content/rso-protein-design-exploration
```

## Step 7 — First: run 100 aa reproduction

Execute `notebooks/01_rso_reproduction.ipynb` cell by cell.

What happens per stage cell:
- **Stage 1 (RSO):** JAX compiles the first step (1–5 min) then runs 100 design iterations. 5–15 min for 100 aa on A100.
- **Stage 2 (ProteinMPNN):** very fast — typically < 1 min, downloads ProteinMPNN weights on first run.
- **Stage 3 (Validation):** 8 × AF2 single-sequence model_4_ptm predictions — longest stage, ~30–90 min on A100 for 100 aa. Each candidate saves a metrics JSON and predicted PDB. If a single candidate fails with OOM/error, the remaining candidates still run.

## Step 8 — Check outputs

Open the file browser in Colab (left sidebar). You should see:

```
/content/results/
  runs/<run_id>/
    metadata.json
    status.json          # status.status should become "completed"
    stage1_rso/bb0.pdb
    stage1_rso/bb0_loss_history.csv
    stage2_mpnn/bb0_candidates.fasta
    stage2_mpnn/bb0_candidates.csv
    stage3_validation/
      bb0_c0_predicted.pdb
      bb0_c0_metrics.json
      ... (c1..c7)
```

If all 8 candidates have predicted PDBs + metrics.json and `status.json` says `"status": "completed"`, the reproduction run succeeded computationally.

## Step 9 — Then run length experiment (quick)

Upload or open `notebooks/02_length_experiment.ipynb`. In the config cell set `PROFILE='quick'` (this picks `configs/length_experiment_quick.yaml` — lengths 100,150,200 × 1 backbone each). Expect total runtime of several hours on A100.

Use 'full' profile only with budget for a longer session.

## Step 10 — Download the results zip

The final cells of each notebook write a file named `rso_results_YYYYmmddTHHMMSSZ.zip` to `/content/`. The `files.download()` call triggers your browser to download it. Save it to your Downloads folder.

If the download is interrupted or too large (big archive), move the zip to Drive first via `!cp /content/rso_results_*.zip /content/drive/MyDrive/` and then download via Google Drive web UI.

## Step 11 — Merge results zip back into your local repo

On your local machine (macOS/Linux/WSL):

```bash
cd /path/to/rso-protein-design-exploration
# assumes you downloaded to ~/Downloads/
unzip -o ~/Downloads/rso_results_*.zip -d results/
# verify it looks right
ls results/runs/ results/metrics/
```

The zip already stores files under the `runs/` and `metrics/` sub-paths, so unzipping directly into `results/` matches the repository layout.

## Step 12 — Run verify_outputs.py locally

```bash
PYTHONPATH=src python3 scripts/verify_outputs.py --require-real-metrics
```

Exit 0 = pass. Errors/warnings are printed with `ERROR:` / `WARN:` prefixes.

## Step 13 — Generate figures

```bash
PYTHONPATH=src python3 scripts/make_figures.py
```

Lists PNG files actually written. 0 figures written is expected if no numeric metrics exist yet (we do not produce synthetic figures).

## Step 14 — Update README Results sections

Open `README.md`. The "Results" and "Observations" sections ship in "Pending execution" state. Only edit them after `verify_outputs.py --require-real-metrics` passes. Use exact numbers from `results/metrics/all_candidates.csv` and `backbone_summary.csv`. Avoid overclaiming; write sample size explicitly ("n = 3 backbones × 8 candidates = 24 sequences total").

## Step 15 — Commit & push to GitHub

Standard git workflow:

```bash
git status
git add -p README.md docs/ results/metrics/ figures/ notebooks/ configs/ scripts/ src/ tests/
git commit -m "Add 100 aa reproduction results + length experiment quick"
git push origin main
```

Always double-check staged diffs do not include large weights, secrets, or unintended large zips. `.gitignore` should catch most of them, but `git status` is the ground truth.
