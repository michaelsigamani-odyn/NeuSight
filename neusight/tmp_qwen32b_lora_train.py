import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


@dataclass
class TrainConfig:
    model_name: str
    data_path: str
    output_dir: str
    num_epochs: float
    learning_rate: float
    batch_size: int
    grad_accum: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    max_length: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=10)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(row: Dict[str, str]) -> str:
    instruction = row.get("instruction") or row.get("prompt") or ""
    input_text = row.get("input") or ""
    return instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"


def build_response(row: Dict[str, str]) -> str:
    return row.get("output") or row.get("response") or row.get("answer") or ""


def render_text(tokenizer: AutoTokenizer, row: Dict[str, str]) -> str:
    messages = [{"role": "user", "content": build_prompt(row)}, {"role": "assistant", "content": build_response(row)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def make_dataset(cfg: TrainConfig, tokenizer: AutoTokenizer) -> Dataset:
    rows = read_rows(cfg.data_path)
    texts = [render_text(tokenizer, row) for row in rows]
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=cfg.max_length), batched=True, remove_columns=["text"])


def lora_targets() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model(cfg: TrainConfig) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=lora_targets())
    lora_model = get_peft_model(model, lora)
    lora_model.config.use_cache = False
    lora_model.enable_input_require_grads()
    return lora_model


def build_args(cfg: TrainConfig) -> TrainingArguments:
    return TrainingArguments(output_dir=cfg.output_dir, bf16=True, num_train_epochs=cfg.num_epochs, learning_rate=cfg.learning_rate, per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum, logging_steps=5, save_strategy="epoch", report_to="none", dataloader_num_workers=4, gradient_checkpointing=True, ddp_find_unused_parameters=False)


def build_tokenizer(cfg: TrainConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = "right"
    return tokenizer


def train() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg)
    dataset = make_dataset(cfg, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=build_args(cfg), train_dataset=dataset, data_collator=collator)
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    train()
