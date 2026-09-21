import torch
from torch.profiler import profile, ProfilerActivity
import pandas as pd
from pathlib import Path
import pandas as pd
import json
import ast
import torch
import numpy as np
import os
from neusight.Prediction.predictor import MLPredictor
import json
import pandas as pd

def profile_bmm(B, M, N, K):
    def measure_once(m, x, starter, ender):
        starter.record()
        out = m(*x)
        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender)

    active = 25
    mean = 5

    def profile_time(m,x,starter,ender,active,mean):
        assert(active > mean)

        # time
        lat_list = []

        # warmup
        lat_list.append(measure_once(m,x,starter,ender)) 

        for _ in range(active):
            lat_list.append(measure_once(m,x,starter,ender))
        lat_list.sort()
        lat_list = lat_list[:mean]
        lat = sum(lat_list)/len(lat_list)

        return lat


    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    dtype = torch.float16

    x = torch.randn(B, M, N, device="cuda").to(dtype)
    y = torch.randn(B, N, K, device="cuda").to(dtype)

    lat = profile_time(torch.bmm,[x,y],starter,ender,active,mean)
    return lat

entries = []

for N in [256, 512, 1024]:
    for B in [128, 256, 512]:
        lat = profile_bmm(B, N, N, N)
        entries.append({"B":B, "M":N, "N":N, "K":N, "Latency":lat})

df = pd.DataFrame(entries)

predictor_path = Path("data/predictor/MLP_WAVE")
bmm_tile_dataset = Path("data/dataset/train/bmm.csv")

bmm_predictor = MLPredictor(predictor_path/"BMM", meta_table_path=bmm_tile_dataset)

with open("./data/device_configs/NVIDIA_H100_80GB_HBM3_TP.json") as f:
    device_config = json.load(f)

df["pred"] = df.apply(lambda x: bmm_predictor.predict(["bmm"], {"B" : x["B"], "M": x["M"], "N": x["N"], "K": x["K"]}, device_config=device_config) * 1000, axis=1)

df["err"] = abs(df["pred"] - df["Latency"])/df["Latency"]

df.to_csv("fp16mm.csv")

print(df)