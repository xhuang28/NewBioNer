#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP15_RestartFalse_Comb1 N21_0.9649_0.9603_0.9697_15 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP15_RestartFalse_Comb1 N21_0.9649_0.9603_0.9697_15 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP4_RestartFalse_Comb1 N21_LAST_0.8655_0.8592_0.8719_4 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP4_RestartFalse_Comb1 N21_LAST_0.8655_0.8592_0.8719_4 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP68_RestartFalse_Comb1 N21_LAST_0.8833_0.8457_0.9243_68 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP68_RestartFalse_Comb1 N21_LAST_0.8833_0.8457_0.9243_68 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP32_RestartFalse_Comb3 N21_0.9746_0.9677_0.9816_32 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP32_RestartFalse_Comb3 N21_0.9746_0.9677_0.9816_32 pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb3 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb3 ?? pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP8_RestartFalse_Comb4 N21_LAST_0.5250_0.4772_0.5835_8 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP8_RestartFalse_Comb4 N21_LAST_0.5250_0.4772_0.5835_8 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP27_RestartFalse_Comb4 N21_LAST_0.8286_0.7716_0.8946_27 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP27_RestartFalse_Comb4 N21_LAST_0.8286_0.7716_0.8946_27 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP2_RestartFalse_Comb5 N21_LAST_0.9625_0.9569_0.9682_2 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP2_RestartFalse_Comb5 N21_LAST_0.9625_0.9569_0.9682_2 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP1_RestartFalse_Comb5 N21_0.9299_0.9601_0.9016_1 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP1_RestartFalse_Comb5 N21_0.9299_0.9601_0.9016_1 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP40_RestartFalse_Comb6 N21_LAST_0.5355_0.4872_0.5944_40 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP40_RestartFalse_Comb6 N21_LAST_0.5355_0.4872_0.5944_40 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP40_RestartFalse_Comb6 N21_0.8920_0.8431_0.9469_40 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP40_RestartFalse_Comb6 N21_0.8920_0.8431_0.9469_40 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP24_RestartFalse_Comb7 N21_0.9705_0.9616_0.9797_24 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP24_RestartFalse_Comb7 N21_0.9705_0.9616_0.9797_24 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP39_RestartFalse_Comb7 N21_0.9761_0.9709_0.9813_39 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP39_RestartFalse_Comb7 N21_0.9761_0.9709_0.9813_39 pickle2

sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb8 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb8 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb8 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb8 ?? pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP1_RestartFalse_Comb9 N21_0.9639_0.9552_0.9727_1 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP1_RestartFalse_Comb9 N21_0.9639_0.9552_0.9727_1 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP1_RestartFalse_Comb9 N21_0.9505_0.9436_0.9575_1 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP1_RestartFalse_Comb9 N21_0.9505_0.9436_0.9575_1 pickle2

sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb10 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P32_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb10 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 U P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb10 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_global.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP??_RestartFalse_Comb10 ?? pickle2



squeue -u huan183