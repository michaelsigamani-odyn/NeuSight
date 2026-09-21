import neusight
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--predictor_name", type=str, help="Name of the predictor")
parser.add_argument("--predictor_path", type=str, help="Path to the predictor")

parser.add_argument("--device_config_path", type=str, help="Path to the device config")
parser.add_argument("--model_config_path", type=str, help="Path to the model config")
parser.add_argument("--sequence_length", type=int, help="Sequence length")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--execution_type", type=str, help="Execution type")

parser.add_argument("--tile_dataset_dir", type=str, help="Path to the tile dataset directory")
parser.add_argument("--result_dir", type=str, help="Path to the result directory")

parser.add_argument("--options", type=str, help="Options", default="")

parser.add_argument("--running_device", type=str, help="Options", default=None)
parser.add_argument("--use_lora", action="store_true", help="Enable LoRA-wrapped tracing in training mode")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated target modules")
parser.add_argument("--lora_trace_with_peft", action="store_true", help="Trace PEFT-injected base model instead of synthetic LoRA approximation")

args = parser.parse_args()

import os
if args.running_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.running_device.split(":")[1]

# initialize neusight predictor
neusight_predictor = neusight.NeusightPredictor(
    predictor_name=args.predictor_name,
    predictor_path=args.predictor_path,
    tile_dataset_dir=args.tile_dataset_dir,
)

# make prediction
neusight_predictor.predict(
    device_config_path=args.device_config_path,
    model_config_path=args.model_config_path,
    sequence_length=args.sequence_length,
    batch_size=args.batch_size,
    execution_type=args.execution_type,
    result_dir=args.result_dir,
    options=args.options,
    use_lora=args.use_lora,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    lora_target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    lora_trace_with_peft=args.lora_trace_with_peft,
)
