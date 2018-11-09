#!/bin/bash

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
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



C7D="$EVAL_FOLDER/BioNLP13CG-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 7 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 7 "$C1D"


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 8 "$C7D"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 8 "$C7D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 9 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 9 "$C1D"


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 10 "$C7D"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 10 "$C7D"



C11D="$EVAL_FOLDER/CRAFT-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 11 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 11 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 12 "$C11D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 12 "$C11D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 13 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 13 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 14 "$C11D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 14 "$C11D"



C15D="$EVAL_FOLDER/CELLFINDER-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 15 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 15 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 17 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 17 "$C1D"



C19D="$EVAL_FOLDER/CHEMPROT-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 19 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 19 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P32 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 21 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp1.sh P33 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 M nosig 100 300 False 21 "$C1D"

