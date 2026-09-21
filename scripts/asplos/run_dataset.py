from pathlib import Path
from neusight.Dataset.collect import dataset_merge
from neusight.Dataset.collect import vec_merge
from pathlib import Path
from neusight.Dataset.collect import collect
from neusight.Dataset.collect import *
import neusight
import sys


# Dataset
base_dir = Path("data/dataset_amd")
# base_dir = Path("data/dataset_amd")

# ln and softmax
for ops in ["ln", "softmax"]:
    dataset_merge(base_dir/"train/collect"/ops, base_dir/"train/collect"/(ops+".csv"))
ln_merge(base_dir/"train/collect", base_dir/"train"/"ln.csv")
softmax_merge(base_dir/"train/collect", base_dir/"train"/"softmax.csv")

# elem
elem_ops = ["add", "mul", "pow", "div",] + ["addu", "mulu", "powu", "divu", "relu", "gelu", "tanh",]
for ops in elem_ops:
    dataset_merge(base_dir/"train/collect"/ops, base_dir/"train/collect"/(ops+".csv"))
elem_merge(Path(base_dir/"train/collect"), base_dir/"train/elem.csv")

# bmm
dataset_merge(Path(base_dir/"train/collect/bmm"), base_dir/"train/bmm.csv")
# dataset_merge(Path(base_dir/"test/collect/bmm"), base_dir/"test/bmm.csv")

# linear
dataset_merge(Path(base_dir/"train/collect/linear"), base_dir/"train/linear.csv")
# dataset_merge(Path(base_dir/"test/collect/linear"), base_dir/"test/linear.csv")

# vec
files = [base_dir/"train/ln.csv", base_dir/"train/softmax.csv", base_dir/"train/elem.csv",]
dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)
df_merged = pd.concat(dfs)
df_merged.to_csv(base_dir/"train/vec.csv", index=False)

# habitat
files = [base_dir/"train/ln.csv", base_dir/"train/softmax.csv", base_dir/"train/elem.csv",]
dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)
df_merged = pd.concat(dfs)
df_merged.to_csv(base_dir/"train/vec.csv", index=False)
