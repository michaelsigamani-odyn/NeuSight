from neusight.Tracing.trace import trace_graph
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
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--seqlen", type=int, required=True)
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--overwrite", type=str2bool, default=True)
parser.add_argument("--single_layer", type=str2bool, default=True)
args = parser.parse_args()

model_config_path = args.model_config_path
mode = args.mode
batch_size = args.batch_size
seqlen = args.seqlen
overwrite = args.overwrite

model_name = Path(model_config_path).name
model_name = model_name.split(".")[0] # remove .json

# model_name = "switch3b"
# mode = "train"
# batch_size = 1
# seqlen = 512
# overwrite = True

is_train = (mode == "train")
single_layer = not ("switch" in model_name)

# paths
device_name = torch.cuda.get_device_name(device=None)
if device_name == "AMD Instinct MI250X / MI250":
    device_name = "AMD Instinct MI250"
device_name = device_name.replace(" ", "_")

model_desc = f"{model_name}-{mode}-{seqlen}-{batch_size}"
trace_name = Path(args.result_dir)/f"{device_name}/{model_desc}.csv"

if os.path.exists(trace_name) and not overwrite:
    print("skipping trace ", trace_name)
else:
    df, e2e_dict = trace_graph(model_config_path=model_config_path, sequence_length=seqlen, batch_size=batch_size, is_train=is_train, bench=True, single_layer=single_layer)
    if df is not None:
        dump_df(df, trace_name)
        dump_json(e2e_dict, Path(str(trace_name).replace(".csv", ".json")))
    else:
        print("trace failed")
print("")