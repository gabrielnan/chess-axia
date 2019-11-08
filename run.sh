#!/usr/bin/env bash

python train.py \
    --batch-size 128 \
    --shuffle \
    --lr 0.01 \
    --num-test 1 \
    --epochs 50 \
    --log-interval 1


