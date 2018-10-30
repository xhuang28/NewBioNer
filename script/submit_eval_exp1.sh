#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2
#sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P1 U P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P1 M P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 U P22_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse N21_0.8477_0.8504_0.8451_9 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse N21_0.8477_0.8504_0.8451_9 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 U P22_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartTrue N21_0.8439_0.8435_0.8443_48 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartTrue N21_0.8439_0.8435_0.8443_48 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 U P23_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse N21_0.8477_0.8569_0.8386_24 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse N21_0.8477_0.8569_0.8386_24 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 U P23_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartTrue N21_0.8405_0.8396_0.8414_22 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval_exp1.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartTrue N21_0.8405_0.8396_0.8414_22 pickle2





squeue -u huan183

