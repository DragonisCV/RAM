opt1='options/5task/5task_pretrain.yaml'
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/train.py \
--auto_resume \
-opt $opt1 \
--launcher pytorch 