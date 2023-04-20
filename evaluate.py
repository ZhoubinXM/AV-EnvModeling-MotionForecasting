import argparse
import yaml
from train_eval.evaluator import Evaluator
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters",
                    default="./configs/deepfm_subgraph_globalgraph_mlp.yml")
parser.add_argument("-r", "--data_root", help="Root directory with data",
                    required=False)
parser.add_argument("-d", "--data_dir", help="Directory to extract data",
                    required=False)
parser.add_argument("-o", "--output_dir", help="Directory to save results",
                    default="./result")
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint",
                    default="./output/checkpoints/best.tar")
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'results')):
    os.mkdir(os.path.join(args.output_dir, 'results'))


# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


# Evaluate
evaluator = Evaluator(cfg, args.data_root, args.data_dir, args.checkpoint)
evaluator.evaluate(output_dir=args.output_dir)
