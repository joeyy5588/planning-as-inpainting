import yaml
from box import Box
import argparse
import os
from datasets import build_dataset
from models import build_model, build_text_encoder
from eval import build_evaluator
from trainer import build_scheduler, build_train_loop
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import notebook_launcher
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/decision.yaml', help='Path to the config file.')
parser.add_argument('--output_dir', type=str, default='output/sample', help='Path to the output directory.')
args = parser.parse_args()

# Load config file
config = (yaml.safe_load(open(args.config)))

# Save config file
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)

config = Box(config)
model_config = config.model
training_config = config.training
training_config.output_dir = args.output_dir
dataset_config = config.dataset

train_dataset = build_dataset(dataset_config)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config.train_batch_size, shuffle=True)
# print(train_dataset[0].trajectories[:-1].shape, train_dataset[0].trajectories[-1].shape, train_dataset[0].conditions[0].shape, train_dataset[0].instructions)
# quit()
val_dataset_config = dataset_config.copy()
val_dataset_config.max_n_episodes = val_dataset_config.max_n_episodes // 10
if "Kuka" in dataset_config.name:
    val_dataset_config.train = True
else:
    val_dataset_config.train = False
val_dataset = build_dataset(val_dataset_config)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=training_config.eval_batch_size, shuffle=True)
evaluator = build_evaluator(dataset_config)

model, ema_model = build_model(model_config)
noise_scheduler = build_scheduler(training_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

if "lr_warmup_rates" in training_config:
    lr_warmup_rates = training_config.lr_warmup_rates
else:
    lr_warmup_rates = 0.05

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=(len(train_dataloader) * training_config.num_epochs * lr_warmup_rates),
    num_training_steps=(len(train_dataloader) * training_config.num_epochs),
)

# print(model)
train_loop = build_train_loop(dataset_config)

if not hasattr(model_config, 'text_encoder_name'):
    train_loop(training_config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, evaluator, lr_scheduler, ema_model)

else:
    text_encoder, tokenizer = build_text_encoder(model_config)
    train_loop(training_config, model, text_encoder, tokenizer, noise_scheduler, optimizer, train_dataloader, val_dataloader, evaluator, lr_scheduler, ema_model)


