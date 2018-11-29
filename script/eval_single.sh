#!/bin/bash

source activate bioner

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

PHASE=$1
EXEC_NAME=$3
CHECKPOINT_NAME=$4

SRC_FOLDER="/media/storage_e/npeng/bioner/xiao/github/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/eval"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints"


cd $SRC_FOLDER

python3 -u $SRC_FOLDER/eval_single.py \
  --load_arg $LOAD_CHECKPOINT/$PHASE/EXP2/$EXEC_NAME/$CHECKPOINT_NAME.json \
  --load_check_point $LOAD_CHECKPOINT/$PHASE/EXP2/$EXEC_NAME/$CHECKPOINT_NAME.model \
  --local_eval \
  --pickle 0 \
  --pred_method $2 \
  --test_file $SRC_FOLDER/corpus/train/$5/test.tsv \
  | tee $SRC_FOLDER/logs/$PHASE/eval_$2._$EXEC_NAME.log

source deactivate
