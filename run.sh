#!/usr/bin/env bash

python train.py \
    --batch-size 150 \
    --shuffle \
    --lr 0.0004 \
    --num-test 1000 \
    --init-epoch 0 \
    --epochs 100 \
    --log-interval 500 \
    --save-interval 1000 \
    --eval-interval 500 \
    --num-gpus 1 \
    #--model-loadname models/axia000.pt \
    #--ae-model models/ae_small.pt \
    #--boards-file data/boards_small.npz \



