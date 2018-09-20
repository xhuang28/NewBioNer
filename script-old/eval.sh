#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8


EXP_NAME="EXP1"
EXEC_NAME="EXP1"
SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/eval"
LOGS_FOLDER="$SRC_FOLDER/logs"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints"


cd $SRC_FOLDER

python3 $SRC_FOLDER/eval.py \
  --load_arg $LOAD_CHECKPOINT/EXP1/N21_0.8900_0.8444_0.9407_12.json \
  --load_check_point $LOAD_CHECKPOINT/EXP1/N21_0.8900_0.8444_0.9407_12.model \
  --if_pred \
  --pred_file \
  $DATA_FOLDER/CELLFINDER-IOBES/test.tsv \
  $DATA_FOLDER/BioNLP13CG-IOBES/test.tsv \
  $DATA_FOLDER/CHEMPROT-IOBES/test.tsv \
  | tee $LOGS_FOLDER/e$EXEC_NAME.log

source deactivate
