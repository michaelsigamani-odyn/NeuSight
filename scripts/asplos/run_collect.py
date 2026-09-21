from pathlib import Path
from neusight.Dataset.collect import collect
from neusight.Dataset.collect import *
import neusight
import sys

base_dir = Path("data/dataset_amd")

# bmm
neusight.generate_train_bmm(base_dir/"dims/train_bmm.csv")
neusight.generate_test_bmm("data/dataset_amd/dims/test_bmm.csv")
collect(opname="bmm", dims_path=base_dir/"dims/train_bmm.csv", out_dir_base=base_dir/"train")
collect(opname="bmm", dims_path=base_dir/"dims/test_bmm.csv", out_dir_base=base_dir/"test")
# linear
neusight.generate_train_linear(base_dir/"dims/train_linear.csv")
neusight.generate_test_linear("data/dataset_amd/dims/test_linear.csv")
collect(opname="linear", dims_path=base_dir/"dims/train_linear.csv", out_dir_base=base_dir/"train")
collect(opname="linear", dims_path=base_dir/"dims/test_linear.csv", out_dir_base=base_dir/"test")
# ln
neusight.generate_train_ln(base_dir/"dims/train_ln.csv")
collect(opname="ln", dims_path=base_dir/"dims/train_ln.csv", out_dir_base=base_dir/"train")
# softmax
neusight.generate_train_vec(base_dir/"dims/train_vec.csv")
collect(opname="softmax", dims_path=base_dir/"dims/train_vec.csv", out_dir_base=base_dir/"train")
# elem
neusight.generate_train_vec(base_dir/"dims/train_vec.csv")
elem_ops = ["add", "mul", "pow", "div",] + ["addu", "mulu", "powu", "divu", "relu", "gelu", "tanh",]
for opname in elem_ops:
    collect(opname=opname, dims_path=base_dir/"dims/train_vec.csv", out_dir_base=base_dir/"train")