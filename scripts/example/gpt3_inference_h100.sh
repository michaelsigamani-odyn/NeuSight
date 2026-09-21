#!/bin/bash
cd "$(dirname "$0")"

VENV_PY="../../.venv/bin/python"

"${VENV_PY}" ../pred.py \
    --predictor_name neusight \
    --predictor_path ../asplos/data/predictor/MLP_WAVE \
    --device_config_path '../asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json' \
    --model_config_path ../asplos/data/DLmodel_configs/gpt3_27.json \
    --sequence_length 2048 \
    --batch_size 2 \
    --execution_type inf \
    --tile_dataset_dir ../asplos/data/dataset/train \
    --result_dir ./out \
