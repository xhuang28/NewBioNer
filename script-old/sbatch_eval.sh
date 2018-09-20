#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=24:00:00 --mem=32000 ./eval.sh

squeue -n eval.sh

