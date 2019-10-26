#!/bin/bash

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

PHASE=$1
EXEC_NAME=$3
CHECKPOINT_NAME=$4

SRC_FOLDER="."
DATA_FOLDER="$SRC_FOLDER/corpus/eval"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints"


cd $SRC_FOLDER

python3 -u $SRC_FOLDER/eval.py \
  --load_arg $LOAD_CHECKPOINT/$PHASE/EXP2/$EXEC_NAME/$CHECKPOINT_NAME.json \
  --load_check_point $LOAD_CHECKPOINT/$PHASE/EXP2/$EXEC_NAME/$CHECKPOINT_NAME.model \
  --if_pred \
  --pred_file \
  $DATA_FOLDER/BioNLP13CG-IOBES/test.tsv \
  $DATA_FOLDER/BC5CDR-IOBES/test.tsv \
  $DATA_FOLDER/BioNLP11ID-IOBES/test.tsv \
  --local_eval \
  --pickle $5 \
  --pred_method $2 \
  | tee $SRC_FOLDER/logs/$PHASE/eval_$2._$EXEC_NAME.log
