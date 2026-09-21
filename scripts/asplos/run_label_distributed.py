import subprocess
import torch
import os
import sys
from pathlib import Path

jobs = []

def launch_task(model_name, batch_size, seqlen,  dp_degree, pp_degree, pp_num_microbatch, tp_degree, use_profiler=False, blocking=False,):
    model_config_path = Path.home() / Path("NeuSight/scripts/asplos/data/DLmodel_configs")/(model_name+".json")

    p = subprocess.Popen(
        " ".join([
            "accelerate launch",
            "--multi_gpu",
            "--use_megatron_lm",
            f"--num_processes={dp_degree*pp_degree*tp_degree}",
            "--num_machines=1",
            "--mixed_precision=no",
            "--dynamo_backend=no",
            "--megatron_lm_gradient_clipping", "1.0",
            "--megatron_lm_tp_degree", str(tp_degree),
            "--megatron_lm_pp_degree", str(pp_degree),
            "--megatron_lm_num_micro_batches", str(pp_num_microbatch),
            "--megatron_lm_sequence_parallelism", "false",
            "--megatron_lm_recompute_activations", "false",
            "--megatron_lm_use_distributed_optimizer", "false",
            "./collect_label_distributed.py",
            "--model_config_path", str(model_config_path),
            "--dp_degree", str(dp_degree),
            "--tp_degree", str(tp_degree),
            "--pp_degree", str(pp_degree),
            "--pp_num_microbatch", str(pp_num_microbatch),
            "--global_batch_size", str(batch_size),
            "--seqlen", str(seqlen),
            "--result_dir", "./label",
            "--use_profiler", str(use_profiler),
        ]),
        shell=True,
        stdout=sys.stdout, 
        stderr=sys.stdout, 
        env=os.environ
    )

    if blocking:
        p.wait()

    jobs.append(p)

def trace():
    launch_task(model_name="gpt2_large", batch_size=4, seqlen=1024, blocking=True, dp_degree=4, pp_degree=1, pp_num_microbatch=1, tp_degree=1)
    launch_task(model_name="gpt2_large", batch_size=4, seqlen=1024, blocking=True, dp_degree=1, pp_degree=4, pp_num_microbatch=1, tp_degree=1)
    launch_task(model_name="gpt2_large", batch_size=4, seqlen=1024, blocking=True, dp_degree=1, pp_degree=1, pp_num_microbatch=1, tp_degree=4)

    launch_task(model_name="gpt2_large", batch_size=16, seqlen=1024, blocking=True, dp_degree=4, pp_degree=1, pp_num_microbatch=1, tp_degree=1)
    launch_task(model_name="gpt2_large", batch_size=16, seqlen=1024, blocking=True, dp_degree=1, pp_degree=4, pp_num_microbatch=1, tp_degree=1)
    launch_task(model_name="gpt2_large", batch_size=16, seqlen=1024, blocking=True, dp_degree=1, pp_degree=1, pp_num_microbatch=1, tp_degree=4)

    launch_task(model_name="gpt3_xl", batch_size=4, seqlen=2048, blocking=True, dp_degree=4, pp_degree=1, pp_num_microbatch=1, tp_degree=1)
    launch_task(model_name="gpt3_xl", batch_size=4, seqlen=2048, blocking=True, dp_degree=1, pp_degree=4, pp_num_microbatch=1, tp_degree=1)
    launch_task(model_name="gpt3_xl", batch_size=4, seqlen=2048, blocking=True, dp_degree=1, pp_degree=1, pp_num_microbatch=1, tp_degree=4)

    for p in jobs:
        p.wait()

def main():
    trace()

try:
    main()
except Exception as e:
    print(e)
    for p in jobs:
        print(p.stdout)
        p.kill()