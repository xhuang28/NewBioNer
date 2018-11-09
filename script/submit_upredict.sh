#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./upredict_exp1.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./upredict_exp2.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./upredict_exp1.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./upredict_exp2.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1.sh P1/EXP1/P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 nosig 100 400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2.sh P1/EXP2/P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 nosig 100 400

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1.sh P1/EXP1/P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 nosig 100 400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2.sh P1/EXP2/P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 nosig 100 400

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1.sh P1/EXP1/P12_M_nosig_C100_W400_MV0.5 N21_0.8493_0.8547_0.8440_57 nosig 100 400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2.sh P1/EXP2/P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 nosig 100 400

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1.sh P1/EXP1/P10_M_nosig_C100_W300_MV0 N21_0.8016_0.8513_0.7574_51 nosig 100 300 1
# haven't run:
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2.sh P1/EXP2/P10_M_nosig_C100_W300_MV0_EP51 ?? nosig 100 300 1

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1.sh P2/EXP1/P22_P10_M_nosig_C100_W300_MV0_M_nosig_C100_W300_RestartTrue_IT1 N21_0.8038_0.8618_0.7532_112 nosig 100 300 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2.sh P2/EXP2/P22_P10_M_nosig_C100_W300_MV0_M_nosig_C100_W300_RestartTrue_IT1_EP?? ?? nosig 100 300 2
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1.sh P2/EXP1/P23_P10_M_nosig_C100_W300_MV0_M_nosig_C100_W300_RestartTrue_IT1 N21_0.8033_0.8622_0.7519_46 nosig 100 300 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2.sh P2/EXP2/P23_P10_M_nosig_C100_W300_MV0_M_nosig_C100_W300_RestartTrue_IT1_EP?? ?? nosig 100 300 2



squeue -u huan183