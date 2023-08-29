import argparse
import yaml
from train_eval.evaluator import Evaluator
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c",
                    "--config",
                    help="Config file with dataset parameters",
                    default="./configs/2d_img.yml")
parser.add_argument("-r",
                    "--data_root",
                    help="Root directory with data",
                    required=False)
parser.add_argument("-d",
                    "--data_dir",
                    help="Directory to extract data",
                    required=False)
parser.add_argument("-o",
                    "--output_dir",
                    help="Directory to save results",
                    default="./output/mlp_resmlp_2_3_full/figs/no_pic")
parser.add_argument("-w",
                    "--checkpoint",
                    help="Path to pre-trained or intermediate checkpoint",
                    default="./output/mlp_resmlp_2_3_full/checkpoints/79.tar")
args = parser.parse_args()

# Make directories
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'results')):
    os.mkdir(os.path.join(args.output_dir, 'results'))
if not os.path.isdir(os.path.join(args.output_dir, 'figs')):
    os.mkdir(os.path.join(args.output_dir, 'figs'))

# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)

# Evaluate
evaluator = Evaluator(cfg, args.data_root, args.data_dir, args.checkpoint)
evaluator.evaluate(output_dir=args.output_dir)
