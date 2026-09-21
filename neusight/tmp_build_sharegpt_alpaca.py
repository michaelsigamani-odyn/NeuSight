import argparse
import json
import random
from dataclasses import dataclass
from typing import List

from datasets import load_dataset


@dataclass
class Example:
    instruction: str
    input: str
    output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="HuggingFaceH4/Bespoke-Stratos-17k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def as_pairs(conversations: List[dict]) -> List[Example]:
    rows: List[Example] = []
    for i in range(len(conversations) - 1):
        first = conversations[i]
        second = conversations[i + 1]
        if first.get("from") in {"human", "user"} and second.get("from") in {"gpt", "assistant"}:
            rows.append(Example(first.get("value", "").strip(), "", second.get("value", "").strip()))
    return [row for row in rows if row.instruction and row.output]


def extract(record: dict) -> List[Example]:
    conversations = record.get("conversations") or []
    return as_pairs(conversations)


def write_rows(path: str, rows: List[Example]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_id, split=args.split)
    dataset = dataset.shuffle(seed=args.seed)
    rows: List[Example] = []
    for item in dataset:
        rows.extend(extract(item))
        if len(rows) >= args.max_samples:
            break
    write_rows(args.output_path, rows[: args.max_samples])
    print(f"wrote {min(len(rows), args.max_samples)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
