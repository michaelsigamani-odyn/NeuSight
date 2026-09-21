import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, default_data_collator
from accelerate import Accelerator
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
from pathlib import Path
import sys
import gc
import json
import os
from accelerate.utils import MegatronLMPlugin
import argparse

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

class DummyDataset(Dataset):
    def __init__(self, sequence_length, dataset_size=1000):
        self.sequence_length = sequence_length
        self.dataset_size = dataset_size
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        x = torch.ones((self.sequence_length,))
        m = torch.ones((self.sequence_length,))
        y = torch.ones((self.sequence_length,))
        return {'input_ids': x, 'attention_mask': m, 'labels': y}

def load_model(model_path, device="cuda"):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return model.to(device)

def dump_json(dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

def prof(
    model_path, 
    global_batch_size,
    dp_degree,
    tp_degree,
    pp_degree,
    pp_num_microbatch,
    sequence_length,
    use_profiler,
    trace_dir,
    json_name
):
    # Configuration
    active = 50 # Number of repetitions
    sample = 10   # Number of least samples to consider

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize Accelerator
    accelerator = Accelerator(MegatronLMPlugin(
        train_iters=100,
        other_megatron_args={"DDP_impl":"torch", "DDP-impl":"torch", "--DDP-impl":"torch"},
    ))
    accelerator.wait_for_everyone()

    # Access the default Megatron-LM arguments
    megatron_lm_default_args = accelerator.state.megatron_lm_plugin.megatron_lm_default_args

    # Update the paths to your vocab and merge files
    megatron_lm_default_args.update({
        "vocab_file": "./vocab.json",
        "merge_file": "./merges.txt",
        "train_iters": 100,
    })

    # Assign the updated arguments back
    accelerator.state.megatron_lm_plugin.megatron_lm_default_args = megatron_lm_default_args

    # Proceed with model initialization
    model = load_model(model_path, device="cuda")
    model = model.train()

    # Calculate batch sizes
    dp_degree = accelerator.num_processes // (accelerator.state.megatron_lm_plugin.tp_degree * accelerator.state.megatron_lm_plugin.pp_degree)
    tp_degree = accelerator.state.megatron_lm_plugin.tp_degree
    pp_degree = accelerator.state.megatron_lm_plugin.pp_degree
    num_micro_batches = accelerator.state.megatron_lm_plugin.num_micro_batches
    per_device_batch_size = global_batch_size // dp_degree // num_micro_batches

    # Prepare data and model
    train_dataset = DummyDataset(sequence_length)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        collate_fn=default_data_collator, 
        batch_size=per_device_batch_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Setup profiling
    measured_times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    import contextlib

    # Configure profiler only if enabled
    profiler = None
    if use_profiler and accelerator.is_main_process:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        def trace_handler(trace_dir):
            def _trace_handler(prof):
                prof.export_chrome_trace(f"{trace_dir}/trace{_trace_handler.step}.json")
                _trace_handler.step += 1
            _trace_handler.step = 0
            return _trace_handler
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
            on_trace_ready=trace_handler(trace_dir=trace_dir),
            profile_memory=False,
            record_shapes=False,
            with_stack=True
        )
    else:
        profiler = contextlib.nullcontext()

    # Training loop with conditional profiling
    completed_steps = 0


    with profiler as p:
        while completed_steps < active:
            for batch in train_dataloader:
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

                start.record()
                
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                end.record()
                end.synchronize()

                optimizer.step()
                optimizer.zero_grad()
                
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    measured_times.append(start.elapsed_time(end))
                    if use_profiler:
                        p.step()

                completed_steps += 1
                if completed_steps >= active:
                    break

    # Save results
    if accelerator.is_main_process:
        measured_times.sort()
        measured_times = measured_times[:sample]
        avg_time = sum(measured_times) / len(measured_times)
            
        # Create result dictionary
        result_dict = {
            "num_layer": 0,
            "e2e_latency": avg_time,
            "fw_latency": 0,
            "bwall_latency": 0,
        }
        dump_json(result_dict, json_name)
        
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
parser.add_argument("--dp_degree", type=int, required=True)
parser.add_argument("--tp_degree", type=int, required=True)
parser.add_argument("--pp_degree", type=int, required=True)
parser.add_argument("--pp_num_microbatch", type=int, required=True)
parser.add_argument("--global_batch_size", type=int, required=True)
parser.add_argument("--seqlen", type=int, required=True)
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--use_profiler", type=str2bool, default=False)

args = parser.parse_args()

# only one type of degree
if args.dp_degree > 1:
    args.tp_degree == 1
    args.pp_degree == 1
    option = f"dp{args.dp_degree}"
elif args.tp_degree > 1:
    args.dp_degree == 1
    args.pp_degree == 1
    option = f"tp{args.tp_degree}"
elif args.pp_degree > 1:
    args.dp_degree == 1
    args.tp_degree == 1
    option = f"pp{args.pp_degree}_{args.pp_num_microbatch}"
else:
    assert(0)

model_name = Path(args.model_config_path).name
model_name = model_name.split(".")[0] # remove .json

# paths
device_name = torch.cuda.get_device_name(device=None)
device_name = device_name.replace(" ", "_")
model_desc = f"{model_name}-{'train'}-{args.seqlen}-{args.global_batch_size}-{option}"
trace_dir = Path(args.result_dir)/f"{device_name}/{model_desc}"
json_name = Path(args.result_dir)/f"{device_name}/{model_desc}.json"

e2e_dict = prof(
    model_path=args.model_config_path, 
    global_batch_size=args.global_batch_size,
    dp_degree=args.dp_degree,
    tp_degree=args.tp_degree,
    pp_degree=args.pp_degree,
    pp_num_microbatch=args.pp_num_microbatch,
    sequence_length=args.seqlen,
    use_profiler=args.use_profiler,
    trace_dir = trace_dir,
    json_name = json_name
)
