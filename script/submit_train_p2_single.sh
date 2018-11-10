#!/bin/bash


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False BC2GM-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False BC2GM-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False BC4CHEMD-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False BC4CHEMD-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False BC5CDR-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False BC5CDR-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False JNLPBA-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False JNLPBA-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False linnaeus-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False linnaeus-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False NCBI-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False NCBI-IOBES



sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False BC2GM-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False BC2GM-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False BC4CHEMD-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False BC4CHEMD-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False BC5CDR-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False BC5CDR-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False JNLPBA-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False JNLPBA-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False linnaeus-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False linnaeus-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False NCBI-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2_single.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 ?? False NCBI-IOBES
