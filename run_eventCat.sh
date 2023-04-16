#!/bin/bash

CMD="python GNN_eventCat.py"

PARAM1=0.001
PARAM2=0.0001
PARAM3=0.00002

TAG1=1e-3
TAG2=1e-4
TAG3=2e-5

CUDA_VISIBLE_DEVICES=0 nohup ${CMD} --lr ${PARAM1} \
    &> ../logs/log_graphnet_eventCat_${TAG1}.txt &
CUDA_VISIBLE_DEVICES=1 nohup ${CMD} --lr ${PARAM2} \
    &> ../logs/log_graphnet_eventCat_${TAG2}.txt &
CUDA_VISIBLE_DEVICES=2 nohup ${CMD} --lr ${PARAM3} \
    &> ../logs/log_graphnet_eventCat_${TAG3}.txt &
