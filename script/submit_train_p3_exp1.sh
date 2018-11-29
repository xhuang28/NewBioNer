#!/bin/bash

SRC_FOLDER="/media/storage_e/npeng/bioner/xiao/github/NewBioNer"
TRAIN_FOLDER="$SRC_FOLDER/corpus/train"
EVAL_FOLDER="$SRC_FOLDER/corpus/eval"

C1D="$TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 1 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 1 "$C1D"


C2D="$EVAL_FOLDER/BioNLP11ID-IOBES/devel.tsv \
    $EVAL_FOLDER/BioNLP13CG-IOBES/devel.tsv \
    $EVAL_FOLDER/CRAFT-IOBES/devel.tsv"

#havn't run:
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 2 "$C2D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 2 "$C2D"



C3D="$EVAL_FOLDER/BioNLP11ID-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 3 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 3 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 4 "$C3D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 4 "$C3D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 5 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 5 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 6 "$C3D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 6 "$C3D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb3 N21_0.8417_0.8630_0.8214_215 M nosig 100 300 False 3 "$C1D" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb6 N21_0.8766_0.8256_0.9342_40 M nosig 100 300 False 6 "$C3D" 2



C7D="$EVAL_FOLDER/BioNLP13CG-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 7 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 7 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 8 "$C7D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 8 "$C7D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 9 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 9 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 10 "$C7D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 10 "$C7D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb7 N21_0.8442_0.8521_0.8365_39 M nosig 100 300 False 7 "$C1D" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb10 N21_0.8339_0.8377_0.8302_42 M nosig 100 300 False 10 "$C7D" 2



C11D="$EVAL_FOLDER/CRAFT-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 11 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 11 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 12 "$C11D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 12 "$C11D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 13 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 13 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 14 "$C11D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 14 "$C11D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb11 N21_0.7989_0.7731_0.8265_87 M nosig 100 300 False 11 "$C1D" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb14 N21_0.7511_0.7836_0.7211_67 M nosig 100 300 False 14 "$C11D" 2



C15D="$EVAL_FOLDER/CELLFINDER-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 15 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 15 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 17 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 17 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb15 N21_0.8457_0.8576_0.8341_115 M nosig 100 300 False 15 "$C1D" 2



C19D="$EVAL_FOLDER/CHEMPROT-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 19 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 19 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 21 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 21 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb19 N21_0.8474_0.8412_0.8537_97 M nosig 100 300 False 19 "$C1D" 2






# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb5_IT2 N21_0.8012_0.8768_0.7376_1 M nosig 100 300 False 5 "$C1D" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb6_IT2 N21_0.8838_0.8393_0.9334_4 M nosig 100 300 False 6 "$C3D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb9_IT2 N21_0.8292_0.8427_0.8161_1 M nosig 100 300 False 9 "$C1D" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb10_IT2 N21_0.8427_0.8453_0.8400_90 M nosig 100 300 False 10 "$C7D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb13_IT2 N21_0.7285_0.7625_0.6974_1 M nosig 100 300 False 13 "$C1D" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb14_IT2 N21_0.7567_0.7926_0.7239_73 M nosig 100 300 False 14 "$C11D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb17_IT2 N21_0.8243_0.8654_0.7870_1 M nosig 100 300 False 17 "$C1D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1_iter.sh P33 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb21_IT2 N21_0.8327_0.8253_0.8403_1 M nosig 100 300 False 21 "$C1D" 3



