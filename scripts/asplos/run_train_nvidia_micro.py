import subprocess
import torch
import os
import sys
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import json
from pathlib import Path

data_dir = Path("data")
(data_dir/"micro").mkdir(parents=True, exist_ok=True)

def linear_regression(df, x_attr, y_attr):
    x = getattr(df,x_attr).values
    y = getattr(df,y_attr).values
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return regr.coef_[0][0], regr.intercept_[0], 

def extract_mm(operator_name):

    def do_micro(device):
        df = pd.read_csv(data_dir/f'dataset/train/{operator_name}.csv')
        df = df.query(f"Device == '{device.replace('_', ' ')}'")
        df = df.copy()
        def add_meta(df):
            df["ops"] = df["B"] * df["M"] * df["N"]* df["K"] * 2
            return df
        df["Latency"] = df["Latency"] * 1e-3 # ms to s
        df = add_meta(df)

        return linear_regression(df, "ops", "Latency")

    device_list = [   
        "NVIDIA_A100-PCIE-40GB",
        "Tesla_V100-PCIE-32GB",
        "Tesla_P100-PCIE-16GB",
        "Tesla_T4",
        "Tesla_P4",
    ]

    # existing gpus
    entries = []
    for dev in device_list:
        print(dev)
        coef, intercept = do_micro(dev)
        dev_name = dev
        with open(data_dir/f'device_configs/{dev_name}.json') as f:
            device_configs = json.load(f)
            bw = device_configs["Mem_Bw"]
            single_flops = device_configs["SingleFLOPs"]
        entry = {"Device": dev, "coef": coef, "intercept": intercept, "BW" : bw, "SingleFLOPs" : single_flops}
        entries.append(entry)

    df = pd.DataFrame(entries)

    # new GPUs
    df["invcoef"] = 1/df["coef"]
    
    coef, intercept = linear_regression(df.iloc[0:5], "BW", "invcoef")

    new_dev_list = [
        "NVIDIA_A100_80GB_PCIe",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_L4",
    ]

    entries = []
    for dev in new_dev_list:
        print(dev)
        with open(data_dir/f'device_configs/{dev}.json') as f:
            device_configs = json.load(f)
            bw = device_configs["Mem_Bw"]
            single_flops = device_configs["SingleFLOPs"]
            entry = {"Device": dev, "coef": 1/(coef * bw + intercept), "intercept": 0, "BW" : bw, "SingleFLOPs" : single_flops}
        entries.append(entry)
    df = pd.concat([df, pd.DataFrame(entries)])
    df = df.reset_index(drop=True)

    df.to_csv(data_dir/f"micro/{operator_name}.csv", index=False)
    return df


def extract_vec(operator_name):
    def do_micro(device):
        df = pd.read_csv(data_dir/f'dataset/train/{operator_name}.csv')
        df = df.query(f"Device == '{device.replace('_', ' ')}'")
        df = df.copy()

        unary_vec_ops  = ["addu", "mulu", "powu", "divu", "relu", "gelu", "tanh",]
        binary_vec_ops = ["add", "mul", "pow", "div",]

        df["unary"] = df["OPName"].apply(lambda x : x in unary_vec_ops)
        df["binary"] = df["OPName"].apply(lambda x : x in binary_vec_ops)
        df["Latency"] = df["Latency"] / 1000

        # unary
        df1 = df.query("unary == True")
        df1 = df1.copy()
        df1["mem"] = df1["B"] * df1["H"] * 4 * 2

        # binary
        df2 = df.query("binary == True")
        df2 = df2.copy()
        df2["mem"] = df2["B"] * df2["H"] * 4 * 3

        # special case for layer norm
        df3 = df.query("OPName == 'ln'")
        df3 = df3.copy()
        df3["mem"] = (df3["B"]*df3["H"]) * 4 * 2 # (mean + var) + (mean var load) + (load_vec + store_vec)

        # special case for softmax
        df4 = df.query("OPName == 'softmax'")
        df4 = df4.copy()
        df4["mem"] = (df4["B"]*df4["H"]) * 4 * 2

        df = pd.concat([df1, df2, df3, df4])
        df = df.copy()

        return linear_regression(df, "mem", "Latency")

    device_list = [   
        "NVIDIA_A100-PCIE-40GB",
        "Tesla_V100-PCIE-32GB",
        "Tesla_P100-PCIE-16GB",
        "Tesla_T4",
        "Tesla_P4",
    ]

    # existing gpus
    entries = []
    for dev in device_list:
        print(dev)
        coef, intercept = do_micro(dev)
        with open(data_dir/f'device_configs/{dev}.json') as f:
            device_configs = json.load(f)
            bw = device_configs["Mem_Bw"]
            single_flops = device_configs["SingleFLOPs"]
        entry = {"Device": dev, "coef": coef, "intercept": intercept, "BW" : bw, "SingleFLOPs" : single_flops}
        entries.append(entry)

    df = pd.DataFrame(entries)

    # new GPUs
    df["invcoef"] = 1/df["coef"]
    coef, intercept = linear_regression(df, "BW", "invcoef")

    new_dev_list = [
        "NVIDIA_A100_80GB_PCIe",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_L4",
    ]

    entries = []
    for dev in new_dev_list:
        print(dev)
        with open(data_dir/f'device_configs/{dev}.json') as f:
            device_configs = json.load(f)
            bw = device_configs["Mem_Bw"]
            entry = {"Device": dev, "coef": 1/(coef * bw + intercept), "intercept": 0, "BW" : bw}
        entries.append(entry)
    df = pd.concat([df, pd.DataFrame(entries)])
    df = df.reset_index(drop=True)

    df.to_csv(data_dir/f"micro/{operator_name}.csv", index=False)

    return df

print("bmm")
df = extract_mm("bmm")
print("linear")
df = extract_mm("linear")
print("elem")
df = extract_vec("elem")
print("ln")
df = extract_vec("ln")
print("softmax")
df = extract_vec("softmax")