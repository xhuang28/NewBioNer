#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="P1/EXP1" # Fix me!

# 1: char hidden size; 2: word hidden size; 3,4: mask values
EXEC_NAME="CONLL_C${1}_W${2}_MV${3}_${4}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/conll2003_converted"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/$FOLDER/$EXEC_NAME"


cd $SRC_FOLDER
mkdir -p $CHECKPOINT_FOLDER

python3 -u $SRC_FOLDER/train_p1.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/glove.6B.200d.txt \
  --train_file \
  $DATA_FOLDER/LOC/train.tsv \
  $DATA_FOLDER/MISC/train.tsv \
  $DATA_FOLDER/ORG/train.tsv \
  $DATA_FOLDER/PER/train.tsv \
  --dev_file \
  $DATA_FOLDER/LOC/devel.tsv \
  $DATA_FOLDER/MISC/devel.tsv \
  $DATA_FOLDER/ORG/devel.tsv \
  $DATA_FOLDER/PER/devel.tsv \
  --test_file \
  $DATA_FOLDER/LOC/devel.tsv \
  $DATA_FOLDER/MISC/devel.tsv \
  $DATA_FOLDER/ORG/devel.tsv \
  $DATA_FOLDER/PER/devel.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid nosig \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch 500 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden $1 \
  --word_hidden $2 \
  --drop_out 0.5 \
  --pickle $SRC_FOLDER/pickle_conll \
  --idea P14 \
  --pred_method M \
  --multi_mask $3 $4 \
  | tee $SRC_FOLDER/logs/$FOLDER/$EXEC_NAME.log


conda deactivate
