#!/bin/bash

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
TRAIN_FOLDER="$SRC_FOLDER/corpus/train"
EVAL_FOLDER="$SRC_FOLDER/corpus/eval"

C1T="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv"
C1C="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"
C1E="$EVAL_FOLDER/CELLFINDER-IOBES/test.tsv \
    $EVAL_FOLDER/BioNLP13CG-IOBES/test.tsv \
    $EVAL_FOLDER/CHEMPROT-IOBES/test.tsv \
    $EVAL_FOLDER/BioNLP11ID-IOBES/test.tsv \
    $EVAL_FOLDER/CRAFT-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 1 "$C1T" "$C1E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 1 "$C1C" "$C1E"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 2 "$C1T" "$C1E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 2 "$C1C" "$C1E"



C3T="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv"
C3C="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"
C3E="$EVAL_FOLDER/BioNLP11ID-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 3 "$C3T" "$C3E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 3 "$C3C" "$C3E"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 4 "$C3T" "$C3E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 4 "$C3C" "$C3E"



C5T="0"
C5C="0"
C5E="$EVAL_FOLDER/BioNLP11ID-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 5 "$C5T" "$C5E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 5 "$C5C" "$C5E"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 6 "$C5T" "$C5E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 6 "$C5C" "$C5E"



C7T="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv"
C7C="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"
C7E="$EVAL_FOLDER/BioNLP13CG-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 7 "$C7T" "$C7E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 7 "$C7C" "$C7E"


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 8 "$C7T" "$C7E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 8 "$C7C" "$C7E"



C9T="0"
C9C="0"
C9E="$EVAL_FOLDER/BioNLP13CG-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 9 "$C9T" "$C9E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 9 "$C9C" "$C9E"


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 10 "$C9T" "$C9E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 10 "$C9C" "$C9E"


C11T="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv"
C11C="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"
C11E="$EVAL_FOLDER/CRAFT-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 11 "$C11T" "$C11E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 11 "$C11C" "$C11E"


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 12 "$C11T" "$C11E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 12 "$C11C" "$C11E"



C13T="0"
C13C="0"
C13E="$EVAL_FOLDER/CRAFT-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 13 "$C13T" "$C13E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 13 "$C13C" "$C13E"


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 14 "$C13T" "$C13E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 14 "$C13C" "$C13E"



C15T="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv"
C15C="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"
C15E="$EVAL_FOLDER/CELLFINDER-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 15 "$C15T" "$C15E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 15 "$C15C" "$C15E"



C17T="0"
C17C="0"
C17E="$EVAL_FOLDER/CELLFINDER-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 17 "$C17T" "$C17E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 17 "$C17C" "$C17E"



C19T="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv"
C19C="$TRAIN_FOLDER/BC2GM-IOBES/train.tsv \
    $TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/train.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/train.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/train.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/train.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/train.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"
C19E="$EVAL_FOLDER/CHEMPROT-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 19 "$C19T" "$C19E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 19 "$C19C" "$C19E"



C21T="0"
C21C="0"
C21E="$EVAL_FOLDER/CHEMPROT-IOBES/test.tsv"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 21 "$C21T" "$C21E"
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 21 "$C21C" "$C21E"



squeue -u huan183