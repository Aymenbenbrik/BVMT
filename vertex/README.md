# Vertex AI Custom Jobs — TFT Ablation Sweep

Infrastructure for running the four Action 1.4 jobs (TFT v3 baseline + three
ablations) on Google Cloud Vertex AI. Each job is a self-contained Python
training script (`training/tft_ablations/runner.py`) packaged in a Docker
image that bundles the BVMT dataset, training code, and pinned dependencies.

| File | Role |
|---|---|
| `Dockerfile` | Vertex AI training image, based on `pytorch-gpu.2-3.py310` |
| `requirements.txt` | Pinned BVMT-specific dependencies (torch comes from the base image) |
| `cloudbuild.yaml` | Cloud Build manifest to build + push the image to Artifact Registry |
| `submit.py` | Submits one or all four Custom Jobs via `gcloud ai custom-jobs create` |

## One-time setup

```bash
# 1. Auth + project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable APIs
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com

# 3. Artifact Registry repo (us-central1; rename if you change region)
gcloud artifacts repositories create bvmt \
    --repository-format=docker \
    --location=us-central1 \
    --description="BVMT TFT training images"

# 4. GCS bucket for run artefacts (replace YOUR_PROJECT_ID)
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-bvmt
```

## Build the image

```bash
gcloud builds submit \
    --config=vertex/cloudbuild.yaml \
    --substitutions=_IMAGE=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/bvmt/tft-ablations:latest \
    .
```

Build takes ~10 min; ~5 GB image because of the CUDA + PyTorch base.

## Submit jobs

```bash
# Single variant (smoke + cost-control friendly)
python vertex/submit.py \
    --variant no_vsn \
    --project YOUR_PROJECT_ID \
    --region us-central1 \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/bvmt/tft-ablations:latest \
    --bucket gs://YOUR_PROJECT_ID-bvmt \
    --accelerator-type NVIDIA_TESLA_T4 \
    --epochs 100

# All four jobs (baseline + 3 ablations)
python vertex/submit.py --variant all ...
```

Add `--dry-run` to print the `gcloud` command without submitting.

## GPU sizing

| Accelerator | Per-job time | Per-job cost (us-central1) | Full-sweep cost |
|---|---|---|---|
| T4   | ~10–12 h | ~$3.5  | ~$15 |
| V100 | ~3–4 h   | ~$10   | ~$35 |
| A100 | ~2 h     | ~$7    | ~$30 |

The runner uses bf16 if supported (A100, L4) and fp16 otherwise (T4, V100).

## Output layout (per job)

```
gs://YOUR_PROJECT_ID-bvmt/<variant>/<timestamp>/
    models/<run_name>_NN_X.YYYY.ckpt
    models/<variant>_best.ckpt
    logs/<run_name>/version_0/metrics.csv
    <variant>_summary.json
```

The `<variant>_summary.json` is what feeds `tab:tft_ablations` in the article
(`paper/bvmt_multiagent_ieee_article.tex` §V.E). Pull it back with
`gsutil cp` after the job finishes.

## Confidence-gating ablation (post-hoc)

The third ablation does NOT need a Vertex job — it operates on the v3
predictions CSV. After the baseline job has finished and its checkpoint is
local, run:

```bash
# 1. Generate the per-row predictions (local GPU or Vertex)
python training/evaluate_tft_v3_per_row.py \
    --ckpt models/tft_bvmt_bestv3.ckpt \
    --out  results/tft_v3_predictions_2025.csv

# 2. Sweep the gating threshold (CPU, ~10 s)
python -m training.tft_ablations.no_gating_eval
```

This produces `results/tft_ablations/confidence_gating_sweep.csv` which
populates the no-gating row of `tab:tft_ablations`.

## Smoke test (no Vertex needed)

Before paying for cluster time, confirm each variant builds and trains a
mini-batch on your local CPU/GPU:

```bash
python -m training.tft_ablations.runner --variant baseline    --smoke
python -m training.tft_ablations.runner --variant no_vsn      --smoke
python -m training.tft_ablations.runner --variant global_norm --smoke
```

Each smoke run takes 2–4 min on CPU and writes a summary JSON to
`results/tft_ablations/<variant>_summary.json`.
