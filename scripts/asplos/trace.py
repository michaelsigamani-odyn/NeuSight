import neusight
import argparse
from pathlib import Path
import os
from neusight.Tracing.trace import trace_graph

def dump_df(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

parser = argparse.ArgumentParser()

parser.add_argument("--model_config_path", type=str, help="Path to the model config")
parser.add_argument("--sequence_length", type=int, help="Sequence length")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--execution_type", type=str, help="Execution type")
parser.add_argument("--result_dir", type=str, help="Path to the result directory")
parser.add_argument("--fusion", action="store_true", help="Fusion", default=False)
args = parser.parse_args()

result_dir = args.result_dir
execution_type = args.execution_type
model_config_path = args.model_config_path
model_name = Path(model_config_path).name.split(".")[0]
sequence_length = args.sequence_length
batch_size = args.batch_size
fusion = args.fusion

result_dir = Path(result_dir)

is_train = (execution_type == "train")

if model_name is None:
    model_name = Path(model_config_path).name.split(".")[0]

# parse options
distributed = False
dp_degree = 1
tp_degree = 1
pp_degree = 1
pp_num_microbatch = 1

if fusion or "switch" in model_name:
    single_layer = False
else:
    single_layer = True

import re

if fusion:
    model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion"
    trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion.csv"
    parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion.csv"
else:
    model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}"
    trace_name = result_dir/f"opgraph_raw/{model_tag}.csv"
    parse_name = result_dir/f"opgraph/{model_tag}.csv"

# trace raw operator graph
print(trace_name)
if os.path.exists(trace_name):
    print("already exists : ", os.path.realpath(trace_name))
    pass
else:
    df, _ = trace_graph(
                        model_config_path=model_config_path, 
                        sequence_length=sequence_length, 
                        batch_size=batch_size, 
                        is_train=is_train, 
                        bench=False, 
                        single_layer=single_layer, 
                        fusion=fusion,
                        distributed=distributed,
                        dp_degree=dp_degree,
                        pp_degree=pp_degree,
                        pp_num_microbatch=pp_num_microbatch,
                        tp_degree=tp_degree,
                    )
    dump_df(df, trace_name)
