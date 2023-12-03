import yaml
from box import Box
import argparse
import os
from datasets import build_dataset
from models import build_model
from eval import build_evaluator
import torch
from diffusers import DDPMScheduler
from accelerate.logging import get_logger
from eval import build_pipeline
from trainer import build_eval_loop

logger = get_logger(__name__, log_level="INFO")


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='output/simple', help='Path to the checkpoint directory.')
args = parser.parse_args()

# Load config file
config = Box(yaml.safe_load(open(args.checkpoint + "/config.yaml")))
model_config = config.model
training_config = config.training
dataset_config = config.dataset

val_dataset_config = dataset_config.copy()
val_dataset_config.max_n_episodes = val_dataset_config.max_n_episodes // 10
val_dataset_config.train = False
val_dataset = build_dataset(val_dataset_config)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=training_config.eval_batch_size, shuffle=True)
evaluator = build_evaluator(dataset_config)

pipeline = build_pipeline(dataset_config, args.checkpoint)

eval_loop = build_eval_loop(dataset_config)

eval_loop(training_config, pipeline, val_dataloader, evaluator)

