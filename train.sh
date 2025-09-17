opt1='options/7task/7task_pretrain.yaml'
opt2='options/7task/7task_mac.yaml'
opt3='options/7task/7task_finetune.yaml'

torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/train.py \
--auto_resume \
-opt $opt1 \
--launcher pytorch 

python scripts/adaSAM_mac_analysis.py \
-opt $opt2 

torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/train.py \
--auto_resume \
-opt $opt3 \
--launcher pytorch 