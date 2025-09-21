torchrun \
--nproc_per_node=[num of gpus] \
--master_port=[PORT] ram/train.py \
-opt [OPT] \
--launcher pytorch
