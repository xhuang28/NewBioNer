#!/bin/bash


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 True

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 True

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 M nosig 100 400 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 M nosig 100 400 True

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 M nosig 100 400 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 M nosig 100 400 True

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 M nosig 100 400 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P12_M_nosig_C100_W400_MV0.3 N21_0.8487_0.8595_0.8382_53 M nosig 100 400 False

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P12_M_nosig_C100_W400_MV0.5 N21_0.8493_0.8547_0.8440_57 M nosig 100 400 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P12_M_nosig_C100_W400_MV0.5 N21_0.8493_0.8547_0.8440_57 M nosig 100 400 False

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P1/EXP1 P10_M_nosig_C100_W300_MV0 N21_0.8016_0.8513_0.7574_51 M nosig 100 300 False 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P1/EXP1 P10_M_nosig_C100_W300_MV0 N21_0.8016_0.8513_0.7574_51 M nosig 100 300 False 1

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P22 P2/EXP1 P22_P10_M_nosig_C100_W300_MV0_M_nosig_C100_W300_RestartTrue_IT1 N21_0.8038_0.8618_0.7532_112 M nosig 100 300 True 2
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1.sh P23 P2/EXP1 P23_P10_M_nosig_C100_W300_MV0_M_nosig_C100_W300_RestartTrue_IT1 N21_0.8033_0.8622_0.7519_46 M nosig 100 300 True 2


squeue -u huan183