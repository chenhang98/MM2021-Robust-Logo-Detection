#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
OUT=$3
GPUS=${GPUS:-8}
PORT=${PORT:-29500}

echo 'checkpoint file' ${CHECKPOINT}
echo 'gpus ' ${GPUS}
echo 'out file' ${OUT_FILE}
echo 'eval method: ' ${EVAL}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=4 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --format-only \
    --options "jsonfile_prefix=${3}"
