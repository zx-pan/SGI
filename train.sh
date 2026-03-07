#!/usr/bin/env bash
set -euo pipefail

export PATH=$PATH:YOUR_PATH/mpeg-pcc-tmc13/build/tmc3

IMAGE_DIR="${1:-}"
OUTPUT_ROOT="${2:-./outputs/default_run}"

if [[ -z "${IMAGE_DIR}" ]]; then
  echo "Error: IMAGE_DIR is required."
  echo "Usage: bash train.sh /path/to/image_dir [output_root]"
  exit 1
fi

python train_multi_scale.py \
  --image_dir "${IMAGE_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --multi_scale_level 3 \
  --feature_dim 50 \
  --n_offsets_num 10 \
  --anchor_number 350000 \
  --eval \
  --lod 0 \
  --voxel_size 0.001 \
  --update_init_factor 4 \
  --iterations 15000 \
  --resolution 1
