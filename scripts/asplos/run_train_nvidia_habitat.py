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
    launch_task(model_config_path="./data/predictor/configs/HABITAT_BMM.json", 
                trainset_path="./data/dataset/train/bmm.csv", 
                save_path="./data/predictor/HABITAT/BMM", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)

    # linear
    launch_task(model_config_path="./data/predictor/configs/HABITAT_LINEAR.json", 
                trainset_path="./data/dataset/train/linear.csv", 
                save_path="./data/predictor/HABITAT/LINEAR", 
                log_dir="./out/logs", 
                epochs=100, 
                blocking=False)

    # vec
    launch_task(model_config_path="./data/predictor/configs/HABITAT_VEC.json", 
                trainset_path="./data/dataset/train/elem.csv", 
                save_path="./data/predictor/HABITAT/VEC", 
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