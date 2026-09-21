import neusight
import argparse
import os
import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument("--model_config_path", type=str, help="Path to the model config")
parser.add_argument("--trainset_path", type=str, help="Path to the trainset")
parser.add_argument("--save_path", type=str, help="Path to save the model")
parser.add_argument("--log_dir", type=str, help="Path to save the logs")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--device", type=str, help="GPU device to train on", default="cuda:0")


args = parser.parse_args()

torch.cuda.set_device(args.device)

model = neusight.model_provider(args.model_config_path)
trainer = neusight.Trainer(model, save_path=args.save_path, log_dir=args.log_dir)

trainer.train(trainset_path=args.trainset_path, epochs=args.epochs)

print("train finished")
print("model saved at", os.path.realpath(args.save_path))