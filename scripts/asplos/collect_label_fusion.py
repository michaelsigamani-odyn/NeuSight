from neusight.Tracing.trace import measure_e2e
import argparse
from pathlib import Path
import torch
import os
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dump_df(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def dump_json(dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("--model_config_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--seqlen", type=int, required=True)
parser.add_argument("--result_dir", type=str, required=True)
args = parser.parse_args()

model_config_path = args.model_config_path
batch_size = args.batch_size
seqlen = args.seqlen

model_name = Path(model_config_path).name
model_name = model_name.split(".")[0]

# paths
device_name = torch.cuda.get_device_name(device=None)
device_name = device_name.replace(" ", "_")

model_desc = f"{model_name}-{'inf'}-{seqlen}-{batch_size}-{'fusion'}"
json_file = Path(args.result_dir)/f"{device_name}/{model_desc}.json"

e2e_dict = measure_e2e(model_config_path=model_config_path, sequence_length=seqlen, batch_size=batch_size, is_train=False, fusion=True)
dump_json(e2e_dict, json_file)
