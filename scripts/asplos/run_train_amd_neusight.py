import subprocess
import torch
import os
import sys

def cuda_rr():
  idx = cuda_rr.cuda_idx % cuda_rr.device_num
  cuda_rr.cuda_idx += 1
  return "cuda:"+str(idx)
cuda_rr.cuda_idx = 0
cuda_rr.device_num = torch.cuda.device_count()

jobs = []

def launch_task(model_config_path, trainset_path, save_path, log_dir, epochs, blocking=False):
    p = subprocess.Popen(
        [
            "python3", "../train.py",
            "--model_config_path", model_config_path,
            "--trainset_path", trainset_path,
            "--save_path", save_path,
            "--log_dir", log_dir,
            "--epochs", str(epochs),
            "--device", cuda_rr(),
        ],
        stdout=sys.stdout, 
        stderr=sys.stdout, 
    )

    if blocking:
        p.wait()

    jobs.append(p)

def train():

    # bmm
    launch_task(model_config_path="./data/predictor/configs/MLP_WAVE_BMM_AMD.json", 
                trainset_path="./data/dataset_amd/train/bmm.csv", 
                save_path="./data/predictor/MLP_WAVE_AMD/BMM", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)

    # linear
    launch_task(model_config_path="./data/predictor/configs/MLP_WAVE_LINEAR_AMD.json", 
                trainset_path="./data/dataset_amd/train/linear.csv", 
                save_path="./data/predictor/MLP_WAVE_AMD/LINEAR", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)

    # ln
    launch_task(model_config_path="./data/predictor/configs/MLP_WAVE_LN_AMD.json", 
                trainset_path="./data/dataset_amd/train/ln.csv", 
                save_path="./data/predictor/MLP_WAVE_AMD/LN", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)
    
    # softmax
    launch_task(model_config_path="./data/predictor/configs/MLP_WAVE_SOFTMAX_AMD.json", 
                trainset_path="./data/dataset_amd/train/softmax.csv", 
                save_path="./data/predictor/MLP_WAVE_AMD/SOFTMAX", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)
    
    # vec
    launch_task(model_config_path="./data/predictor/configs/MLP_WAVE_VEC_AMD.json", 
                trainset_path="./data/dataset_amd/train/elem.csv", 
                save_path="./data/predictor/MLP_WAVE_AMD/VEC", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)

    # wait for processes
    for p in jobs:
        p.wait()

def main():
    train()

try:
    main()
except Exception as e:
    print(e)
    for p in jobs:
        print(p.stdout)
        p.kill()