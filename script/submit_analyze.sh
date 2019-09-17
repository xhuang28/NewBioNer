#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N50/N21_0.9579_0.9671_0.9488_70 0 1        50
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N100/N21_0.9669_0.9703_0.9635_68 0 1        100
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N150/N21_0.9656_0.9709_0.9603_81 0 1        150
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N200/N21_0.9662_0.9676_0.9648_69 0 1        200
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N250/N21_0.9597_0.9615_0.9579_68 0 1        250
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N300/N21_0.9674_0.9707_0.9642_69 0 1        300
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N350/N21_0.9648_0.9689_0.9606_68 0 1        350
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV0_1_N400/N21_0.9648_0.9675_0.9621_68 0 1        400
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N50/N21_0.9065_0.9327_0.8817_5   1 1        50
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N100/N21_0.9140_0.9256_0.9027_6   1 1        100
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N150/N21_0.9142_0.9346_0.8946_5   1 1        150
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N200/N21_0.9164_0.9350_0.8985_5   1 1        200
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N250/N21_0.9110_0.9334_0.8896_8   1 1        250
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N300/N21_0.9147_0.9327_0.8973_8   1 1        300
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N350/N21_0.9144_0.9229_0.9061_6   1 1        350
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./analyze.sh FT_conll2003_converted_MV1_1_N400/N21_0.9114_0.9357_0.8884_6   1 1        400
