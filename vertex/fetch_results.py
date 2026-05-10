"""vertex/fetch_results.py — pull TFT ablation summaries from GCS.

After a Vertex AI Custom Job completes, the runner uploads the
contents of /app/results/tft_ablations to AIP_MODEL_DIR (a GCS path
that submit.py builds as <bucket>/<variant>/<timestamp>/model). This
script walks each variant subdirectory, finds the most recent
summary JSON, and copies it back to the local
results/tft_ablations/ tree. It does NOT download the large
checkpoint files (.ckpt) by default --- pass --with-ckpts to also
fetch them.

USAGE
    python vertex/fetch_results.py --bucket gs://my-bucket
    python vertex/fetch_results.py --bucket gs://my-bucket --with-ckpts

PRE-REQUISITES
    pip install google-cloud-storage
    gcloud auth application-default login (or attached service account)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True,
                   help="GCS bucket prefix the jobs wrote to "
                        "(e.g. gs://my-bucket).")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "tft_ablations",
                   help="Local directory to copy summaries into.")
    p.add_argument("--variants", nargs="+",
                   default=["baseline", "no_vsn", "global_norm"],
                   help="Variant subdirectories to fetch.")
    p.add_argument("--with-ckpts", action="store_true",
                   help="Also download .ckpt files (large; tens of MB each).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        print("ERROR: google-cloud-storage not installed. "
              "Run: pip install google-cloud-storage")
        return 1

    if not args.bucket.startswith("gs://"):
        print(f"ERROR: --bucket must start with gs://. Got: {args.bucket}")
        return 1

    rest = args.bucket[5:]
    bucket_name, _, prefix = rest.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    print(f"Bucket : gs://{bucket_name}/")
    print(f"Prefix : {prefix or '<root>'}")
    print(f"Out    : {args.out_dir}")
    print()

    for variant in args.variants:
        variant_prefix = f"{prefix.rstrip('/')}/{variant}/" if prefix else f"{variant}/"
        print(f"--- {variant} ---")
        # Find the most recent timestamp folder under this variant
        blobs = list(client.list_blobs(bucket, prefix=variant_prefix))
        if not blobs:
            print(f"  no objects under gs://{bucket_name}/{variant_prefix}")
            continue
        # Group by timestamp folder (3rd path component after variant)
        timestamps = sorted({b.name.split("/")[2] for b in blobs
                             if len(b.name.split("/")) > 2}, reverse=True)
        if not timestamps:
            print("  no timestamp subdirs found")
            continue
        latest_ts = timestamps[0]
        print(f"  latest run : {latest_ts}")

        target = args.out_dir / variant
        target.mkdir(parents=True, exist_ok=True)
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) < 3 or parts[2] != latest_ts:
                continue
            rel = "/".join(parts[3:])
            if not rel:
                continue
            if not args.with_ckpts and rel.endswith(".ckpt"):
                continue
            local = target / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local))
            kb = blob.size / 1024 if blob.size else 0
            print(f"  -> {local.relative_to(args.out_dir)}  ({kb:.0f} KB)")

        # Pretty-print the variant summary if present
        summary_path = target / f"{variant}_summary.json"
        if not summary_path.exists():
            # try the file written under model/ subdir
            for cand in target.rglob(f"{variant}_summary.json"):
                summary_path = cand
                break
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            acc = s.get("best_q50_dir_acc")
            vl = s.get("best_val_loss")
            ep = s.get("epochs_trained")
            tn = s.get("target_normalizer")
            vp = s.get("vsn_patched")
            print(f"  best_val_loss     : {vl}")
            print(f"  best_q50_dir_acc  : {acc}")
            print(f"  epochs_trained    : {ep}")
            print(f"  target_normalizer : {tn}")
            print(f"  vsn_patched       : {vp}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
