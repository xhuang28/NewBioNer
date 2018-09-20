#!/bin/bash

sbatch --ntasks=1 --partition=isi --time=24:00:00 --mem-per-cpu=24GB --gres=gpu:1 ./make_prediction.sh

squeue -n make_prediction.sh