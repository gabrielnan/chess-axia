#!/usr/bin/env bash

python train.py \
    --batch-size 400 \
    --shuffle \
    --lr 0.0004 \
    --num-test 1000 \
    --init-epoch 0 \
    --epochs 40 \
    --init-iter 168500 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 500 \
    --num-gpus 4 \
    --boards-file data/boards1.npz \
    #--model-loadname models/effe/axia_4_168000.pt \
    #--experiment effe00401c624705b4dfda9271d5349e \
    #--ae-model models/ae_small.pt \



