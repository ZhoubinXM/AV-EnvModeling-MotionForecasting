#! /bin/bash
# run in the top folder.
python -m torch.distributed.launch --master_port 60202 --nproc_per_node 8 --use_env ./train.py -c ./configs/motionformer_v2.yml -o ./output/motionformer_v2_version_4/ --multi_gpu --main_device 0 
