#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8


EXP_NAME="EXP2"
EXEC_NAME="EXP2"
SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/eval"
LOGS_FOLDER="$SRC_FOLDER/logs"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints"


cd $SRC_FOLDER

python3 $SRC_FOLDER/eval.py \
  --load_arg $LOAD_CHECKPOINT/EXP2/N21_0.9419_0.9106_0.9754_51.json \
  --load_check_point $LOAD_CHECKPOINT/EXP2/N21_0.9419_0.9106_0.9754_51.model \
  --if_pred \
  --pred_file \
  $DATA_FOLDER/CELLFINDER-IOBES/test.tsv \
  $DATA_FOLDER/BioNLP13CG-IOBES/test.tsv \
  $DATA_FOLDER/CHEMPROT-IOBES/test.tsv \
  | tee $LOGS_FOLDER/e$EXEC_NAME.log

source deactivate
