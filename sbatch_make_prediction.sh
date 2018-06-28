#!/bin/bash

sbatch --ntasks=1 --partition=isi --time=72:00:00 --mem-per-cpu=32GB --gres=gpu:1 ./make_prediction.sh

squeue -n make_prediction.sh