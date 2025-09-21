torchrun \
--nproc_per_node=1 \
--master_port=[PORT] ram/test.py \
-opt [OPT] --launcher pytorch