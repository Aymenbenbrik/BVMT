# Vertex AI Custom Training image for BVMT TFT ablations
#
# Base: Google's PyTorch + CUDA prebuilt image. Includes torch and
# common DL libs, so we only layer the BVMT-specific deps on top.
# pytorch-forecasting==1.7.0 expects torch >= 2.0 which the base
# image provides; if the base image's torch is older or newer, lock
# torch in vertex/requirements.txt and rebuild.
#
# Build (Cloud Build):  gcloud builds submit --config=vertex/cloudbuild.yaml .
# Run    (Vertex AI):   python vertex/submit.py --variant no_vsn

FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest

WORKDIR /app

# System deps (CSV manipulation, gcsfuse for direct bucket reads if
# desired). Keep this minimal — Cloud Build cache invalidates the
# whole layer on changes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps — copy first for layer caching, install separately
COPY vertex/requirements.txt /app/vertex/requirements.txt
RUN pip install --no-cache-dir -r /app/vertex/requirements.txt

# The pytorch-gpu Vertex base image ships torch_xla (TPU runtime). Lightning
# auto-discovers it and tries to initialise PJRT, which segfaults when no
# PJRT_DEVICE is set. We use CUDA, so the cleanest fix is to remove XLA so
# the discovery never fires. -y is required because the image runs apt as
# root (we never sudo) and || true keeps the layer green if torch_xla is
# already absent in a future base image refresh.
RUN pip uninstall -y torch_xla pytorch-xla 2>/dev/null || true
ENV PJRT_DEVICE=CUDA

# Repo (training scripts + data + agents). The dataset
# data/features/tft_features.csv is checked into the repo and is
# small enough (~30 MB) to ship inside the container; remove this
# COPY and download from GCS at runtime if it grows past ~100 MB.
COPY data/features /app/data/features
COPY training      /app/training
COPY agents        /app/agents
COPY tests         /app/tests
COPY paper/improvement_plan.md /app/paper/improvement_plan.md

ENV PYTHONUNBUFFERED=1 \
    BVMT_SEED=0 \
    PYTHONPATH=/app

# The Vertex AI Custom Job specifies the entrypoint args via
# `python_module` + `args` (see vertex/submit.py). Default ENTRYPOINT
# is the runner so a manual `docker run` works for local debugging.
ENTRYPOINT ["python", "-m", "training.tft_ablations.runner"]
CMD ["--variant", "baseline"]
