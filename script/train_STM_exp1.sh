#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="P1/EXP1" # Fix me!

# 1: Domain name
EXEC_NAME="STM_${1}_C${2}_W${3}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/$FOLDER/$EXEC_NAME"


cd $SRC_FOLDER
mkdir -p $CHECKPOINT_FOLDER
mkdir -p $SRC_FOLDER/pickle/STM_${1}
mkdir -p $SRC_FOLDER/logs/$FOLDER/STM_eval
mkdir -p $SRC_FOLDER/logs/$FOLDER/STM_train

python3 -u $SRC_FOLDER/train_p1.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file \
  $DATA_FOLDER/${1}/train.tsv \
  --dev_file \
  $DATA_FOLDER/${1}/devel.tsv \
  --test_file \
  $DATA_FOLDER/${1}/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid nosig \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch 500 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden $2 \
  --word_hidden $3 \
  --drop_out 0.5 \
  --pickle $SRC_FOLDER/pickle/STM_${1} \
  --idea P10 \
  --pred_method M \
  --mask_value 0 \
  | tee $SRC_FOLDER/logs/$FOLDER/$EXEC_NAME.log


conda deactivate
