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


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb6 N21_0.8766_0.8256_0.9342_40 nosig 100 300 6 "$C5T" "$C5E" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP40_RestartFalse_Comb6 N21_0.8920_0.8431_0.9469_40 nosig 100 300 6 "$C5C" "$C5E" 2



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
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 7 "$C7T" "$C7E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 7 "$C7C" "$C7E"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 8 "$C7T" "$C7E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 8 "$C7C" "$C7E"



C9T="0"
C9C="0"
C9E="$EVAL_FOLDER/BioNLP13CG-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 9 "$C9T" "$C9E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 9 "$C9C" "$C9E"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 10 "$C9T" "$C9E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 10 "$C9C" "$C9E"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb10 N21_0.8339_0.8377_0.8302_42 nosig 100 300 10 "$C9T" "$C9E" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP42_RestartFalse_Comb10 N21_0.8410_0.8516_0.8307_42 nosig 100 300 10 "$C9C" "$C9E" 2



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
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 11 "$C11T" "$C11E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 11 "$C11C" "$C11E"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 12 "$C11T" "$C11E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 12 "$C11C" "$C11E"



C13T="0"
C13C="0"
C13E="$EVAL_FOLDER/CRAFT-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 13 "$C13T" "$C13E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 13 "$C13C" "$C13E"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 14 "$C13T" "$C13E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 14 "$C13C" "$C13E"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb14 N21_0.7511_0.7836_0.7211_67 nosig 100 300 14 "$C13T" "$C13E" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP67_RestartFalse_Comb14 N21_LAST_0.7622_0.7901_0.7363_67 nosig 100 300 14 "$C13C" "$C13E" 2



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
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 15 "$C15T" "$C15E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 15 "$C15C" "$C15E"



C17T="0"
C17C="0"
C17E="$EVAL_FOLDER/CELLFINDER-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 17 "$C17T" "$C17E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 17 "$C17C" "$C17E"




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
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 19 "$C19T" "$C19E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 19 "$C19C" "$C19E"



C21T="0"
C21C="0"
C21E="$EVAL_FOLDER/CHEMPROT-IOBES/test.tsv"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP1 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.8487_0.8528_0.8446_49 nosig 100 300 21 "$C21T" "$C21E"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3.sh P3/EXP2 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 nosig 100 300 21 "$C21C" "$C21E"






sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb3 N21_0.8417_0.8630_0.8214_215 nosig 100 300 3 "$C3T" "$C3E" 2
sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP257_RestartFalse_Comb3 N21_0.9943_0.9933_0.9954_214 nosig 100 300 3 "$C3C" "$C3E" 2

sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb7 N21_0.8442_0.8521_0.8365_39 nosig 100 300 7 "$C7T" "$C7E" 2
sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP39_RestartFalse_Comb7 N21_0.9761_0.9709_0.9813_39 nosig 100 300 7 "$C7C" "$C7E" 2

sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb11 N21_0.7989_0.7731_0.8265_87 nosig 100 300 11 "$C11T" "$C11E" 2
sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP87_RestartFalse_Comb11 N21_LAST_0.9457_0.9119_0.9822_87 nosig 100 300 11 "$C11C" "$C11E" 2

sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb15 N21_0.8457_0.8576_0.8341_115 nosig 100 300 15 "$C15T" "$C15E" 2
sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP115_RestartFalse_Comb15 N21_0.9897_0.9850_0.9945_115 nosig 100 300 15 "$C15C" "$C15E" 2

sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_Comb19 N21_0.8474_0.8412_0.8537_97 nosig 100 300 19 "$C19T" "$C19E" 2
sbatch --partition=isi --time=00:05:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP97_RestartFalse_Comb19 N21_0.9835_0.9730_0.9942_97 nosig 100 300 19 "$C19C" "$C19E" 2







# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb5_IT2 N21_0.8012_0.8768_0.7376_1 nosig 100 300 5 "$C5T" "$C5E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb5_IT2 N21_0.9159_0.9618_0.8742_1 nosig 100 300 5 "$C5C" "$C5E" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb6_IT2 N21_0.8838_0.8393_0.9334_4 nosig 100 300 6 "$C5T" "$C5E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP4_RestartFalse_Comb6_IT2 N21_0.8948_0.8482_0.9469_4 nosig 100 300 6 "$C5C" "$C5E" 3


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb9_IT2 N21_0.8292_0.8427_0.8161_1 nosig 100 300 9 "$C9T" "$C9E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb9_IT2 N21_0.9436_0.9400_0.9473_1 nosig 100 300 9 "$C9C" "$C9E" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb10_IT2 N21_0.8427_0.8453_0.8400_90 nosig 100 300 10 "$C9T" "$C9E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP90_RestartFalse_Comb10_IT2 N21_LAST_0.8487_0.8590_0.8387_90 nosig 100 300 10 "$C9C" "$C9E" 3


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb13_IT2 N21_0.7285_0.7625_0.6974_1 nosig 100 300 13 "$C13T" "$C13E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb13_IT2 N21_0.8565_0.8433_0.8701_1 nosig 100 300 13 "$C13C" "$C13E" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb14_IT2 N21_0.7567_0.7926_0.7239_73 nosig 100 300 14 "$C13T" "$C13E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP73_RestartFalse_Comb14_IT2 N21_LAST_0.7629_0.7951_0.7333_73 nosig 100 300 14 "$C13C" "$C13E" 3


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb17_IT2 N21_0.8243_0.8654_0.7870_1 nosig 100 300 17 "$C17T" "$C17E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb17_IT2 N21_0.9451_0.9541_0.9364_1 nosig 100 300 17 "$C17C" "$C17E" 3


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP1 P3/EXP1 P33_M_nosig_C100_W300_RestartFalse_Comb21_IT2 N21_0.8327_0.8253_0.8403_1 nosig 100 300 21 "$C21T" "$C21E" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./upredict_p3_iter.sh P3/EXP2 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb21_IT2 N21_0.9449_0.9361_0.9539_1 nosig 100 300 21 "$C21C" "$C21E" 3



squeue -u huan183