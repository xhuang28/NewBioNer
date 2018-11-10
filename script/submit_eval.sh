#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P10_M_nosig_C100_W300_MV0_EP51 N21_0.9652_0.9620_0.9683_51 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse N21_0.9618_0.9536_0.9702_9 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse N21_0.9618_0.9536_0.9702_9 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9762_0.9700_0.9826_29 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9762_0.9700_0.9826_29 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P22_P12_M_nosig_C100_W400_MV0.5_EP57_M_nosig_C100_W400_EP14_RestartFalse N21_0.9686_0.9593_0.9781_14 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P22_P12_M_nosig_C100_W400_MV0.5_EP57_M_nosig_C100_W400_EP14_RestartFalse N21_0.9686_0.9593_0.9781_14 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P23_P12_M_nosig_C100_W400_MV0.3_EP53_M_nosig_C100_W400_EP16_RestartFalse N21_0.9711_0.9612_0.9812_16 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W400_MV0.3_EP53_M_nosig_C100_W400_EP16_RestartFalse N21_0.9711_0.9612_0.9812_16 pickle2

# early stopping on single corpus
#BC5DR
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9592_0.9537_0.9649_36 pickle2
#NCBI
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9522_0.9486_0.9559_27 pickle2

#BC5DR
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9601_0.9518_0.9686_5 pickle2
#BC2GM, NCBI
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9607_0.9508_0.9708_7 pickle2
#linnaeus
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_BC5CDR-IOBES_0.9666_0.9643_0.9689_6 pickle2





squeue -u huan183

