#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=2 ./train.py --adadelta --trainRoot traindb --valRoot valdb --cuda"
CUDA_VISIBLE_DEVICES=2 ./train.py --adadelta --trainRoot traindb --valRoot valdb --cuda | tee run.log
