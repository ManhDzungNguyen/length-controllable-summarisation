#!/bin/bash

cd /home2/dungnguyen/length-controllable-summarisation
source /home2/vietle/icgpt/venv/bin/activate
OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=0 python train_bartpho.py
