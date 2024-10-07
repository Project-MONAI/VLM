#! /bin/bash

GPUS=(0 1 2 3 4 5 6 7)
NUM_PARTS=8

for i in ${!GPUS[@]}; do

    export CUDA_VISIBLE_DEVICES=${GPUS[$i]}; python update_texts.py ${GPUS[$i]} ${NUM_PARTS} ${j} &

done

wait
