import os
import sys
import yaml
import shutil
import argparse
import time

from train_eval.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch

from train_eval import logger
from train_eval.utils import get_time, get_color_text
import global_var as gv

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c",
                    "--config",
                    help="Config file with dataset parameters",
                    default="./configs/deepfm_subgraph_globalgraph_mlp.yml")
parser.add_argument("-r",
                    "--data_root",
                    help="Root directory with data",
                    required=False)
parser.add_argument("-d",
                    "--data_dir",
                    help="Directory to extract data",
                    default="./data/extract_data")
parser.add_argument("-o",
                    "--output_dir",
                    help="Directory to save checkpoints and logs",
                    default="./output")
parser.add_argument("-n",
                    "--num_epochs",
                    help="Number of epochs to run training for",
                    default=200)
parser.add_argument("-w",
                    "--checkpoint",
                    help="Path to pre-trained or intermediate checkpoint",
                    default=None)
parser.add_argument("--cuda", help="Use GPU", default=True)
parser.add_argument("--main_device", help="Main device with cuda", default=0)
parser.add_argument("--multi_gpu", help="Use Multi-GPU", action='store_true')
parser.add_argument("--viz",
                    help="Viz attention score in tensorboard.",
                    action='store_true')

args = parser.parse_args()

gv._init()
time_begin = get_time()
gv.set_value('time_begin', time_begin)

logger.info("Parser input param and prepare output dir.")

# Make directories
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.mkdir(os.path.join(args.output_dir, 'checkpoints'))
if not os.path.isdir(os.path.join(args.output_dir, 'saved_model')):
    os.mkdir(os.path.join(args.output_dir, 'saved_model'))
if not os.path.isdir(
        os.path.join(args.output_dir, 'tensorboard_logs', time_begin)):
    os.makedirs(os.path.join(args.output_dir, 'tensorboard_logs', time_begin))

# Save cmd
with open(os.path.join(args.output_dir, 'cmd'), 'w') as file:
    file.write(' '.join(sys.argv))

# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)
cfg['output_dir'] = args.output_dir
cfg['num_epochs'] = args.num_epochs
cfg['use_cuda'] = args.cuda
cfg['main_device'] = args.main_device
cfg['use_multi_gpu'] = args.multi_gpu
cfg['viz'] = args.viz

# Save config
with open(os.path.join(args.output_dir, 'train_args.yaml'),
          mode='w') as dump_dile:
    yaml.dump(cfg, dump_dile)

# Initialize tensorboard writer
writer = SummaryWriter(
    log_dir=os.path.join(args.output_dir, 'tensorboard_logs', time_begin))

# Train
trainer = Trainer(cfg,
                  args.data_root,
                  args.data_dir,
                  checkpoint_path=args.checkpoint,
                  writer=writer)
trainer.train()

# Close tensorboard writer
writer.close()
