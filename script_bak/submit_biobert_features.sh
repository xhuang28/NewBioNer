#!/bin/bash

sbatch --partition=isi --time=12:00:00 --gres=gpu:2 ./biobert_features.sh train/BC2GM-IOBES/train_sents
# sbatch --partition=isi --time=12:00:00 --gres=gpu:2 ./biobert_features.sh train/BC2GM-IOBES/devel_sents
# sbatch --partition=isi --time=12:00:00 --gres=gpu:2 ./biobert_features.sh train/BC2GM-IOBES/test_sents