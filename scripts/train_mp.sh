#! /bin/bash

python -m torch.distributed.launch --nproc_per_nide 8 --use_env train.py -c configs/argoverse.yml -o  -o ./output/test_av_ddp/ --multi_gpu --main_device 0 --viz
