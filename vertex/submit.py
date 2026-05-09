"""vertex/submit.py — submit Vertex AI Custom Training Jobs for the
TFT v3 baseline + the three Action 1.4 ablations.

This wraps `gcloud ai custom-jobs create` so a single command can
launch any one of {baseline, no_vsn, global_norm, all}. The
Dockerfile entrypoint is `python -m training.tft_ablations.runner`,
so all this script does is forward the right CLI args.

USAGE
    # Single ablation
    python vertex/submit.py --variant no_vsn \\
        --project my-gcp-project \\
        --region us-central1 \\
        --image us-central1-docker.pkg.dev/my-gcp-project/bvmt/tft-ablations:latest \\
        --bucket gs://my-bucket/bvmt-runs

    # Submit all four jobs (baseline + three ablations) sequentially
    python vertex/submit.py --variant all ...

PRE-REQUISITES
    1. gcloud auth login                (account with aiplatform.user)
    2. gcloud config set project <ID>
    3. APIs enabled: aiplatform.googleapis.com, artifactregistry.googleapis.com,
       cloudbuild.googleapis.com
    4. GCS bucket created (--bucket flag)
    5. Image built and pushed:
         gcloud builds submit --config=vertex/cloudbuild.yaml .

GPU GUIDANCE
    --accelerator-type NVIDIA_TESLA_T4   ($0.35/h, ~10-12 h per run)
    --accelerator-type NVIDIA_TESLA_V100 ($2.55/h, ~3-4 h per run)
    --accelerator-type NVIDIA_TESLA_A100 ($3.67/h, ~2 h per run)

    Estimated full-sweep cost (4 jobs):
      T4   : ~$15 USD
      V100 : ~$35 USD
      A100 : ~$30 USD
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

VARIANTS = ("baseline", "no_vsn", "global_norm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=(*VARIANTS, "all"), required=True)
    p.add_argument("--project", required=True, help="GCP project ID.")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--image", required=True,
                   help="Full Artifact Registry image URI (must end with :tag).")
    p.add_argument("--bucket", required=True,
                   help="GCS prefix for outputs (e.g. gs://my-bucket/bvmt).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--machine-type", default="n1-standard-8")
    p.add_argument("--accelerator-type", default="NVIDIA_TESLA_T4",
                   choices=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100",
                            "NVIDIA_TESLA_A100", "NVIDIA_L4"])
    p.add_argument("--accelerator-count", type=int, default=1)
    p.add_argument("--service-account", default=None,
                   help="Override service account (defaults to Vertex compute SA).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the gcloud command without executing it.")
    p.add_argument("--seed", type=int, default=0,
                   help="BVMT_SEED env var inside the container.")
    return p.parse_args()


def submit_one(args: argparse.Namespace, variant: str) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"bvmt-tft-{variant.replace('_', '-')}-{timestamp}"
    output_uri = f"{args.bucket.rstrip('/')}/{variant}/{timestamp}"

    worker_pool_spec = {
        "machine_spec": {
            "machine_type": args.machine_type,
            "accelerator_type": args.accelerator_type,
            "accelerator_count": args.accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": args.image,
            "args": [
                "--variant", variant,
                "--epochs", str(args.epochs),
                "--out", "/app/results/tft_ablations",
            ],
            "env": [
                {"name": "BVMT_SEED", "value": str(args.seed)},
                {"name": "AIP_MODEL_DIR", "value": output_uri},
            ],
        },
    }

    config_path = Path(f"/tmp/bvmt_vertex_{variant}_{timestamp}.json")
    config_path.write_text(json.dumps([worker_pool_spec], indent=2))

    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--project={args.project}",
        f"--region={args.region}",
        f"--display-name={job_id}",
        f"--worker-pool-spec=machine-type={args.machine_type},"
        f"replica-count=1,"
        f"accelerator-type={args.accelerator_type},"
        f"accelerator-count={args.accelerator_count},"
        f"container-image-uri={args.image}",
        # Pass runner args via --args so we don't need a JSON config file.
        "--args=--variant=" + variant,
        f"--args=--epochs={args.epochs}",
        "--args=--out=/app/results/tft_ablations",
    ]
    if args.service_account:
        cmd.append(f"--service-account={args.service_account}")

    print(f"\n--- Submitting {variant} ---")
    print("  Job:    ", job_id)
    print("  Region: ", args.region)
    print("  GPU:    ", f"{args.accelerator_type} x{args.accelerator_count}")
    print("  Output: ", output_uri)
    print("  Command:")
    print("    " + " ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        print("  [DRY RUN — not executed]")
        return 0

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("  ERROR: gcloud not found on PATH. Install Google Cloud SDK first.")
        return 2

    if result.returncode != 0:
        print(f"  FAILED ({result.returncode}):")
        print("    stdout:", result.stdout.strip())
        print("    stderr:", result.stderr.strip())
        return result.returncode
    print("  SUBMITTED.")
    print("    " + result.stdout.strip().splitlines()[-1] if result.stdout else "")
    return 0


def main() -> int:
    args = parse_args()
    variants = VARIANTS if args.variant == "all" else (args.variant,)

    rcs = []
    for variant in variants:
        rc = submit_one(args, variant)
        rcs.append(rc)
        # small jitter so monitoring dashboards line up
        if not args.dry_run and len(variants) > 1:
            time.sleep(2)

    failed = sum(1 for rc in rcs if rc != 0)
    if failed:
        print(f"\n{failed}/{len(rcs)} submission(s) failed.")
        return 1
    print(f"\nAll {len(rcs)} job(s) submitted.")
    print("Monitor:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={args.project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
