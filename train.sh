#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=2 ./train.py --adadelta --trainRoot dataset_5w_10/trainset.lmdb --valRoot dataset_5w_10/valset.lmdb --cuda --saveInterval 5000 --valInterval 5000 | tee run.log
