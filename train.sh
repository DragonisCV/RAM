opt1='options/3task/3task_pretrain.yaml'
torchrun \
--nproc_per_node=1 \
--master_port=4321 ram/train.py \
--auto_resume \
-opt $opt1 \
--launcher pytorch 