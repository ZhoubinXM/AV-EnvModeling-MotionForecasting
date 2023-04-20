import os
import yaml
import argparse
import time

from train_eval.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c",
                    "--config",
                    help="Config file with dataset parameters",
                    default="./configs/deepfm_subgraph_globalgraph_mlp.yml")
parser.add_argument("-r", "--data_root", help="Root directory with data", required=False)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", default="./data/extract_data")
parser.add_argument("-o", "--output_dir", help="Directory to save checkpoints and logs", default="./output")
parser.add_argument("-n", "--num_epochs", help="Number of epochs to run training for", default=300)
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", default=None)

args = parser.parse_args()

# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.mkdir(os.path.join(args.output_dir, 'checkpoints'))
if not os.path.isdir(os.path.join(args.output_dir, 'saved_model')):
    os.mkdir(os.path.join(args.output_dir, 'saved_model'))
if not os.path.isdir(os.path.join(args.output_dir, 'tensorboard_logs', time.strftime("%Y%m%d_%H%M%S"))):
    os.mkdir(os.path.join(args.output_dir, 'tensorboard_logs', time.strftime("%Y%m%d_%H%M%S")))

# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)

# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs', time.strftime("%Y%m%d_%H%M%S")))

# Train
trainer = Trainer(cfg, args.data_root, args.data_dir, checkpoint_path=args.checkpoint, writer=writer)
trainer.train(num_epochs=int(args.num_epochs), output_dir=args.output_dir)

# Close tensorboard writer
writer.close()
