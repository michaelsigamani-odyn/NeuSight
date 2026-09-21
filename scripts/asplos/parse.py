import neusight
import argparse
from pathlib import Path
import os
from neusight.Tracing.trace import trace_graph
from neusight.Tracing.parse import parse_trace
import re

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
parser.add_argument("--option", type=str, help="Path to the result directory")
args = parser.parse_args()

result_dir = args.result_dir
execution_type = args.execution_type
model_config_path = args.model_config_path
model_name = Path(model_config_path).name.split(".")[0]
sequence_length = args.sequence_length
batch_size = args.batch_size
options = args.option

is_train = (execution_type == "train")

result_dir = Path(result_dir)

is_train = (execution_type == "train")

if model_name is None:
    model_name = Path(model_config_path).name.split(".")[0]

# parse options
fusion = False
dp_degree = 1
tp_degree = 1
pp_degree = 1
pp_num_microbatch = 1
distributed = False
single_layer = True

if options == "":
    pass
elif options == "fusion":
    fusion = True
    single_layer = False
elif re.match(r"dp\d+", options):
    distributed = True
    dp_degree = int(options[2:])
elif re.match(r"tp\d+", options):
    distributed = True
    tp_degree = int(options[2:])
elif re.match(r"pp\d+_\d+", options):
    distributed = True
    pp_degree = int(options[2:].split("_")[0])
    pp_num_microbatch = int(options[2:].split("_")[1])
else:
    assert(0)

if "switch" in model_name or fusion:
    single_layer = False


if fusion:
    model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion"
    trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion.csv"
    parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-fusion.csv"
elif distributed:
    model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}-{options}"
    if dp_degree > 1:
        trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size // dp_degree}.csv"
        parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-dp{dp_degree}.csv"
    elif tp_degree > 1:
        trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size}.csv"
        parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-tp{tp_degree}.csv"
    elif pp_degree > 1:
        trace_name = result_dir/f"opgraph_raw/{model_name}-{execution_type}-{sequence_length}-{batch_size // pp_num_microbatch}.csv"
        parse_name = result_dir/f"opgraph/{model_name}-{execution_type}-{sequence_length}-{batch_size}-pp{pp_degree}_{pp_num_microbatch}.csv"
else:
    model_tag = f"{model_name}-{execution_type}-{sequence_length}-{batch_size}"
    trace_name = result_dir/f"opgraph_raw/{model_tag}.csv"
    parse_name = result_dir/f"opgraph/{model_tag}.csv"

# parse operator graph
df = parse_trace(
                trace_name, 
                is_train=is_train, 
                bench=False, 
                fusion=fusion,
                distributed=distributed,
                dp_degree=dp_degree,
                pp_degree=pp_degree,
                pp_num_microbatch=pp_num_microbatch,
                tp_degree=tp_degree,
            )

dump_df(df, parse_name)