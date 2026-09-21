import pandas as pd
from pathlib import Path
import math

# Import shared constants
from summarize import (
    PREDICTOR_LIST,
    OOD_MODELS,
    OOD_DEVICES,
    OPERATION_TYPES,
    LATENCY_MODES
)

# Constants
DEVICE_MAP = {
    # NVIDIA GPUs
    "NVIDIA_H100_80GB_HBM3": "H100",
    "NVIDIA_L4": "L4",
    "NVIDIA_A100-PCIE-40GB": "A100-40GB",
    "NVIDIA_A100-SXM4-40GB": "A100x4",
    "Tesla_V100-PCIE-32GB": "V100",
    "Tesla_P100-PCIE-16GB": "P100",
    "Tesla_T4": "T4",
    "Tesla_P4": "P4",
    # AMD GPUs
    "AMD_Instinct_MI100": "MI100",
    "AMD_Instinct_MI210": "MI210",
    "AMD_Instinct_MI250": "MI250",
}

PREDICTOR_MAP = {
    "roofline": "Roofline",
    "habitat": "Habitat",
    "micro": "Li et al.",
    "neusight": "NeuSight",
}

PREDICTOR_LIST = list(PREDICTOR_MAP.keys())

MODEL_CONFIGS = {
    "bert_large": {"name": "BERT-Large", "inf_batch": [8, 16], "train_batch": [2, 8]},
    "gpt2_large": {"name": "GPT2-Large", "inf_batch": [4, 8], "train_batch": [1, 4]},
    "gpt3_xl": {"name": "GPT3-XL", "inf_batch": [2, 8], "train_batch": [1, 2]},
    "opt_13": {"name": "OPT-1.3B", "inf_batch": [2, 8], "train_batch": [1, 2]},
    "gpt3_27": {"name": "GPT3-2.7B", "inf_batch": [2, 8], "train_batch": [1, 2]},
    "switchxl4": {"name": "SwitchTrans", "inf_batch": [1, 2], "train_batch": [1, 2]},
}

# Utility functions
def write_csv(path, rows):
    """Write rows to CSV file"""
    with open(path, 'w') as f:
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')

def format_latency(val):
    """Format latency value"""
    return f"{val:.1f}" if pd.notnull(val) and val != 0 else ""

def format_error(val):
    """Format error value"""
    return f"{val:.2f}" if pd.notnull(val) and val != 0 else ""

def get_summary_value(df, conditions, column):
    """Get value from summary DataFrame"""
    filtered = df.loc[conditions, column]
    return filtered.iloc[0] if not filtered.empty else None

def write_csv_row(file, values, include_comma=True):
    """Standardized CSV row writing"""
    file.write(",".join(str(v) for v in values))
    if include_comma:
        file.write(",")
    file.write("\n")

def get_data_from_summary(summary, condition, tag):
    """Standardized data extraction from summary DataFrame"""
    entry = summary.loc[condition, tag]
    if entry.empty or entry.isna().all():
        return None
    return entry.values[0]

def is_ood(model_name, device):
    if model_name in ["gpt3_27", "opt_13", "gpt3_xl"]:
        return True
    if device in ["NVIDIA_H100_80GB_HBM3", "NVIDIA_L4", "NVIDIA_A100_80GB_PCIe"]:
        return True
    return False

def generate_table(summary, out_path, mode, device_list, headings, predictors, option=None):
    summary = summary.query(f"mode == '{mode}'")
    if option:
        summary = summary.query(f"option == '{option}'")

    with open(out_path, "w") as file:
        for device in device_list:
            file.write(f"{device}\n")

            # Prepare model names and batch sizes
            model_display_names = []
            batch_sizes = []
            for model_key, batches in headings:
                display_name = MODEL_CONFIGS.get(model_key, {}).get("name", model_key)
                model_display_names.extend([display_name] + [""] * (len(batches) - 1))
                batch_sizes.extend(batches)

            # Write model names
            file.write("Model," + ",".join(model_display_names))
            file.write("\n")

            # Write batch sizes
            file.write("Batch," + ",".join(map(str, batch_sizes)))
            file.write("\n")

            # Write data rows
            for predictor in predictors:
                row = [PREDICTOR_MAP.get(predictor, predictor)]
                for model_key, batches in headings:
                    for batch in batches:
                        tag = f"{predictor}_all_e2e_err"
                        condition = (
                            (summary['device'] == device) &
                            (summary['batch'] == batch) &
                            (summary['model_name'] == model_key) &
                            (summary['option'].isnull() if option is None else (summary['option'] == option))
                        )
                        entry = summary.loc[condition, tag]
                        item = format_error(entry.values[0] if not entry.empty else None)
                        row.append(item)
                write_csv_row(file, row, include_comma=False)
            file.write("\n")

def table_nvidia_inf(summary, out_path):
    device_list = [
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_L4",
        "NVIDIA_A100-PCIE-40GB",
        "Tesla_V100-PCIE-32GB",
        "Tesla_P100-PCIE-16GB",
        "Tesla_T4",
        "Tesla_P4",
    ]
    headings = [
        ("bert_large", [8, 16]),
        ("gpt2_large", [4, 8]),
        ("gpt3_xl", [2, 8]),
        ("opt_13", [2, 8]),
        ("gpt3_27", [2, 8]),
        ("switchxl4", [1, 2]),
    ]
    predictors = ["roofline", "habitat", "micro", "neusight"]
    generate_table(summary, out_path, "inf", device_list, headings, predictors)

def table_nvidia_train(summary, out_path):
    device_list = [
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_L4",
        "NVIDIA_A100_80GB_PCIe",
        "Tesla_V100-PCIE-32GB",
    ]
    headings = [
        ("bert_large", [2, 8]),
        ("gpt2_large", [1, 4]),
        ("gpt3_xl", [1, 2]),
        ("opt_13", [1, 2]),
        ("gpt3_27", [1, 2]),
        ("switchxl4", [1, 2]),
    ]
    predictors = ["roofline", "habitat", "micro", "neusight"]
    generate_table(summary, out_path, "train", device_list, headings, predictors)

def table_amd_inf(summary, out_path):
    device_list = [
        "AMD_Instinct_MI100",
        "AMD_Instinct_MI210",
        "AMD_Instinct_MI250",
    ]
    headings = [
        ("bert_large", [8, 16]),
        ("gpt2_large", [4, 8]),
        ("gpt3_xl", [2, 8]),
        ("opt_13", [2, 8]),
        ("gpt3_27", [2, 8]),
    ]
    predictors = ["neusight"]
    generate_table(summary, out_path, "inf", device_list, headings, predictors)

def table_amd_train(summary, out_path):
    device_list = [
        "AMD_Instinct_MI100",
        "AMD_Instinct_MI210",
        "AMD_Instinct_MI250",
    ]
    headings = [
        ("bert_large", [2, 8]),
        ("gpt2_large", [1, 4]),
        ("gpt3_xl", [1, 2]),
        ("opt_13", [1, 2]),
        ("gpt3_27", [1, 2]),
    ]
    predictors = ["neusight"]
    generate_table(summary, out_path, "train", device_list, headings, predictors)

def table_perop(summary, out_path):
    # only nvidia gpus
    file = open(out_path, "w")

    opname_mapping = {
        "BMM": "BMM",
        "Linear": "FC",
        "VEC": "EW",
        "VECsoftmax": "Softmax",
        "VECln": "LN",
    }

    opname_list = ["BMM", "Linear", "VEC", "VECsoftmax", "VECln"]

    file.write("," + "In Distribution," + "," * (len(opname_list)-1) + "Out of Distribution"  + "," * (len(opname_list)-1) + "\n")
    file.write("," + ",".join(list(map(lambda x : opname_mapping[x], opname_list * 2))) + "\n")

    # write numbers
    for predictor in PREDICTOR_LIST:
        file.write(PREDICTOR_MAP.get(predictor, predictor))
        file.write(",")
        for ood in [False, True]:
            for opname in opname_list:
                tag = f"{predictor}_{opname}_e2e_err"
                df = summary[["OOD", tag]]
                df = df.query(f"OOD == {ood}")
                df = df[tag]
                mean = df.mean()
                file.write(str(mean))
                file.write(",")
        file.write("\n")
    file.close()

def table_fusion_inf(summary, out_path):
    summary = summary.query("mode == 'inf'")

    # Separate non-fused and fused summaries
    summary_nonfused = summary[summary['option'].isnull()]
    summary_fused = summary[summary['option'] == 'fusion']

    device_list = [
        "NVIDIA_L4",
        "NVIDIA_A100-PCIE-40GB",
        "NVIDIA_H100_80GB_HBM3",
    ]

    # Prepare headers
    header_devices = []
    for device in device_list:
        header_devices.extend([DEVICE_MAP[device], ""])
    header_nonfused_fused = ["Non-fused", "Fused"] * len(device_list)

    with open(out_path, "w") as file:
        # Write headers
        write_csv_row(file, ["", "", ""] + header_devices, include_comma=False)
        write_csv_row(file, ["", "", ""] + header_nonfused_fused, include_comma=False)

        models = [("BERT", "bert_large"), ("GPT2-Large", "gpt2_large")]
        for model_display, model_name in models:
            batches = [8, 16] if model_name == "bert_large" else [4, 8]
            for batch in batches:
                # Measured Latency
                row = [model_display, f"batch={batch}", "Measured Latency (ms)"]
                for device in device_list:
                    # Collect latencies for non-fused and fused options
                    for summary_df in [summary_nonfused, summary_fused]:
                        condition = (
                            (summary_df['device'] == device) &
                            (summary_df['batch'] == batch) &
                            (summary_df['model_name'] == model_name)
                        )
                        entry = summary_df.loc[condition, 'label_all_e2e_latency']
                        latency = entry.values[0] if not entry.empty else ''
                        row.append(format_latency(latency))
                write_csv_row(file, row, include_comma=False)

                # NeuSight Prediction
                row = ["", "", "NeuSight Prediction (ms)"]
                for device in device_list:
                    # Collect predictions and errors for non-fused and fused options
                    for summary_df in [summary_nonfused, summary_fused]:
                        condition = (
                            (summary_df['device'] == device) &
                            (summary_df['batch'] == batch) &
                            (summary_df['model_name'] == model_name)
                        )
                        pred = summary_df.loc[condition, 'neusight_all_e2e_latency']
                        err = summary_df.loc[condition, 'neusight_all_e2e_err']
                        pred_value = pred.values[0] if not pred.empty else ''
                        err_value = err.values[0] if not err.empty else ''
                        pred_str = (
                            f"{pred_value:.1f} ({err_value:.1f}%)"
                            if pred_value != '' and err_value != '' else ''
                        )
                        row.append(pred_str)
                write_csv_row(file, row, include_comma=False)

def table_distributed(summary, out_path):
    summary = summary.query("mode == 'train'")
    
    device_map = {
        "A100x4": "NVIDIA_A100-SXM4-40GB",
        "H100x4": "NVIDIA_H100_80GB_HBM3"
    }
    parallel_options = ["dp4", "tp4", "pp4_4"]

    with open(out_path, "w") as file:
        # Headers
        write_csv_row(file, ["", "", "", "A100x4", "", "", "H100x4", "", ""])
        write_csv_row(file, ["", "Global batch size", "", "DP", "TP", "PP", "DP", "TP", "PP"])

        last_model = None
        for model, batch in [("GPT2-Large", 4), ("GPT2-Large", 16), ("GPT3-XL", 4)]:
            # Measured Latency
            row = [model if model != last_model else "", batch, "Measured Latency (ms)"]
            for device in device_map:
                for parallel in parallel_options:
                    condition = (
                        (summary['device'] == device_map[device]) &
                        (summary['batch'] == batch) &
                        (summary['model_name'] == model.replace("-", "_").lower()) &
                        (summary['option'] == parallel)
                    )
                    value = get_summary_value(summary, condition, 'label_all_e2e_latency')
                    row.append(format_latency(value))
            write_csv_row(file, row)

            # NeuSight Prediction
            row = ["", "", "NeuSight Prediction (ms)"]
            for device in device_map:
                for parallel in parallel_options:
                    condition = (
                        (summary['device'] == device_map[device]) &
                        (summary['batch'] == batch) &
                        (summary['model_name'] == model.replace("-", "_").lower()) &
                        (summary['option'] == parallel)
                    )
                    pred = get_summary_value(summary, condition, 'neusight_all_e2e_latency')
                    err = get_summary_value(summary, condition, 'neusight_all_e2e_err')
                    if pred is not None and err is not None:
                        row.append(f"{pred:.1f} ({err:.1f}%)")
                    else:
                        row.append("")
            write_csv_row(file, row)

            last_model = model

def main():
    summary_dir = Path("./summary")
    summary = pd.read_csv(summary_dir/"summary.csv")
    
    tables = {
        "nvidia_inf.csv": table_nvidia_inf,
        "nvidia_train.csv": table_nvidia_train,
        "amd_inf.csv": table_amd_inf,
        "amd_train.csv": table_amd_train,
        "perop.csv": table_perop,
        "fusion_inf.csv": table_fusion_inf,
        "distributed.csv": table_distributed
    }

    for filename, table_func in tables.items():
        table_func(summary, summary_dir/filename)

if __name__ == "__main__":
    main()
