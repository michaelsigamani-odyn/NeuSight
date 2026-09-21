import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "content_tokens", "logged_tokens", "load_time_s", "train_time_s", "wall_time_s",
    "content_tok_per_s", "content_tok_per_s_per_gpu", "peak_vram_gb", "power_mean_w",
    "power_peak_w", "energy_j", "tokens_per_joule", "mfu_pct", "status", "oom",
    "host_id", "gpu_uuids", "image_digest", "git_sha",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def invalid_gpu_shape(row: dict[str, str]) -> bool:
    params_b = float(row["params_b"])
    num_gpus = int(row["num_gpus"])
    if params_b <= 24:
        return num_gpus != 1
    if params_b < 70:
        return num_gpus != 2
    return num_gpus != 4


def matrix_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["study"] == "matrix"]


def run_ids_with_bad_gpu_shape(rows: list[dict[str, str]]) -> list[str]:
    return [row["run_id"] for row in rows if invalid_gpu_shape(row)]


def counter_dict(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row[key] for row in rows))


def unique_model_count(rows: list[dict[str, str]]) -> int:
    return len({row["model"] for row in rows})


def validate_tracker_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    matrix = matrix_rows(rows)
    return tracker_report(rows, matrix)


def tracker_report(rows: list[dict[str, str]], matrix: list[dict[str, str]]) -> dict[str, object]:
    return {"total_rows": len(rows), "study_counts": counter_dict(rows, "study"), "status_counts": counter_dict(rows, "status"), "matrix_models": unique_model_count(matrix), "rows_per_model": counter_dict(matrix, "model"), "rows_per_dataset": counter_dict(matrix, "dataset_id"), "rows_per_batch": counter_dict(matrix, "batch_size_per_gpu"), "invalid_gpu_shape_rows": run_ids_with_bad_gpu_shape(matrix)}


def validate_schema(rows: list[dict[str, str]]) -> list[str]:
    return sorted(REQUIRED_COLUMNS - set(rows[0].keys()))


def validate_rates(rows: list[dict[str, str]]) -> dict[str, object]:
    owned = next((row for row in rows if row["rate_id"] == "odyn-owned-a100"), None)
    owned_missing = owned is None or owned["usd_per_gpu_hour"].strip() == ""
    return {
        "rate_ids": [row["rate_id"] for row in rows],
        "owned_rate_missing": owned_missing,
    }


def input_dir() -> Path:
    return Path.home() / "benchmark_inputs"


def report_data(tracker: list[dict[str, str]], rates: list[dict[str, str]]) -> dict[str, object]:
    return validate_tracker_rows(tracker) | {"missing_required_columns": validate_schema(tracker)} | validate_rates(rates)


def main() -> None:
    tracker = load_csv(input_dir() / "finetune_benchmark_tracker_v2.csv")
    rates = load_csv(input_dir() / "rate_table.csv")
    print(json.dumps(report_data(tracker, rates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
