#!/bin/bash


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1_single.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 BC2GM-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2_single.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 BC2GM-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1_single.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 BC4CHEMD-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2_single.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 BC4CHEMD-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1_single.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 BC5CDR-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2_single.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 BC5CDR-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1_single.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 JNLPBA-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2_single.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 JNLPBA-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1_single.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 linnaeus-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2_single.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 linnaeus-IOBES

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp1_single.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 NCBI-IOBES
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./upredict_exp2_single.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 NCBI-IOBES
