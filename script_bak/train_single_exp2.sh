#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="P1/EXP2" # Fix me!

# 1. dataset name
EXEC_NAME="P10_${1}_Epoch${2}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/$FOLDER/$EXEC_NAME"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints/P1/EXP2/P14_M_nosig_C100_W300_MV0_0_EP84/N21_LAST_0.9765_0.9681_0.9849_84"

cd $SRC_FOLDER
mkdir -p $CHECKPOINT_FOLDER
mkdir -p $SRC_FOLDER/pickle/$1

python3 -u $SRC_FOLDER/train_single.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --load_check_point $LOAD_CHECKPOINT.model \
  --load_arg $LOAD_CHECKPOINT.json \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file $DATA_FOLDER/$1/train.tsv \
  --dev_file $DATA_FOLDER/$1/devel.tsv \
  --test_file $DATA_FOLDER/$1/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid nosig \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch ${2} --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden 100 \
  --word_hidden 300 \
  --drop_out 0.5 \
  --pickle $SRC_FOLDER/pickle/$1 \
  --combine \
  --idea P10 \
  --pred_method M \
  --mask_value 0 \
  | tee $SRC_FOLDER/logs/$FOLDER/$EXEC_NAME.log


conda deactivate
