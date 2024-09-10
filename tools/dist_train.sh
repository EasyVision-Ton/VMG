#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
MASTER_PORT=${MASTER_PORT:-29500}

PYTHONPATH="$(dirname $0)/..":PYTHONPATH \
torchrun \
--nproc_per_node $GPUS \
--master_port $MASTER_PORT \
$(dirname $0)/train.py \
--config $CONFIG \
--launcher pytorch ${@:3}

# python -m torch.distributed.launch \
# --nproc_per_node $GPUS \
# --master_port $MASTER_PORT \
# $(dirname $0)/train.py \
# --config $CONFIG \
# --launcher pytorch ${@:3}
