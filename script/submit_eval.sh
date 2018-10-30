#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./eval.sh P1 U P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=6:00:00 ./eval.sh P1 M P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse N21_0.9618_0.9536_0.9702_9 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse N21_0.9618_0.9536_0.9702_9 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9762_0.9700_0.9826_29 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9762_0.9700_0.9826_29 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 pickle2


squeue -u huan183

