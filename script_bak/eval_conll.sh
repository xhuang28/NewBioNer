#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

PHASE=$1
EXEC_NAME=$2
CHECKPOINT_NAME=$3

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/conll2003_converted"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints"


cd $SRC_FOLDER

python3 -u $SRC_FOLDER/eval.py \
  --load_arg $LOAD_CHECKPOINT/$PHASE/EXP2/$EXEC_NAME/$CHECKPOINT_NAME.json \
  --load_check_point $LOAD_CHECKPOINT/$PHASE/EXP2/$EXEC_NAME/$CHECKPOINT_NAME.model \
  --if_pred \
  --pred_file \
  $DATA_FOLDER/test.tsv \
  $DATA_FOLDER/LOC/test.tsv \
  $DATA_FOLDER/PER/test.tsv \
  $DATA_FOLDER/ORG/test.tsv \
  $DATA_FOLDER/MISC/test.tsv \
  --local_eval \
  --pickle pickle_conll2 \
  --pred_method M \
  | tee $SRC_FOLDER/logs/$PHASE/eval_M._$EXEC_NAME.log

source deactivate
