import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import torch
import subprocess

def cuda_rr():
  idx = cuda_rr.cuda_idx % cuda_rr.device_num
  cuda_rr.cuda_idx += 1
  return "cuda:"+str(idx)
cuda_rr.cuda_idx = 0
cuda_rr.device_num = torch.cuda.device_count()

jobs = []

def launch_task(model_name, mode, batch_size, seqlen, device, blocking=True):
    p = subprocess.Popen(
        [
            "python3", "../pred.py",
            "--predictor_name", "neusight",
            "--predictor_path", "./data/predictor/MLP_WAVE_AMD",
            "--device_config_path", f"./data/device_configs/{device}.json",
            "--model_config_path", f"./data/DLmodel_configs/{model_name}.json",
            "--sequence_length", str(seqlen),
            "--batch_size", str(batch_size),
            "--execution_type", mode,
            "--tile_dataset_dir", "./data/dataset_amd/train",
            "--result_dir", "./results",
            "--running_device", cuda_rr(),
        ], 
        stdout=sys.stdout, 
        stderr=sys.stdout, 
    )

    if blocking:
        p.wait()

    jobs.append(p)

def pred():

    # inference
    for device in ["AMD_Instinct_MI100", "AMD_Instinct_MI210", "AMD_Instinct_MI250"]:
        launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, device=device, blocking=False)
        launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt3_xl", mode="inf", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_xl", mode="inf", batch_size=8, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="opt_13", mode="inf", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="opt_13", mode="inf", batch_size=8, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_27", mode="inf", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_27", mode="inf", batch_size=8, seqlen=2048, device=device, blocking=False)

    for p in jobs:
        p.wait()

    # training
    for device in ["AMD_Instinct_MI100", "AMD_Instinct_MI210", "AMD_Instinct_MI250"]:
        launch_task(model_name="bert_large", mode="train", batch_size=2, seqlen=512, device=device, blocking=False)
        launch_task(model_name="bert_large", mode="train", batch_size=8, seqlen=512, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="train", batch_size=1, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt2_large", mode="train", batch_size=4, seqlen=1024, device=device, blocking=False)
        launch_task(model_name="gpt3_xl", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_xl", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="opt_13", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="opt_13", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_27", mode="train", batch_size=1, seqlen=2048, device=device, blocking=False)
        launch_task(model_name="gpt3_27", mode="train", batch_size=2, seqlen=2048, device=device, blocking=False)

    for p in jobs:
        p.wait()

def main():
    pred()

try:
    main()
except Exception as e:
    print(e)
    for p in jobs:
        print(p.stdout)
        p.kill()