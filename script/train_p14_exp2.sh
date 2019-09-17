#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="P1/EXP2" # Fix me!

# 1: char hidden size; 2: word hidden size; 3,4: multiple mask value; 5: # epochs;
EXEC_NAME="$C${1}_W${2}_MV${3}_${4}_EP${5}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/$FOLDER/$EXEC_NAME"


cd $SRC_FOLDER
mkdir -p $CHECKPOINT_FOLDER
mkdir -p $SRC_FOLDER/logs/$FOLDER

python3 -u $SRC_FOLDER/train_p1.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file \
  $DATA_FOLDER/BC2GM-IOBES/train.tsv \
  $DATA_FOLDER/BC4CHEMD-IOBES/train.tsv \
  $DATA_FOLDER/NCBI-IOBES/train.tsv \
  $DATA_FOLDER/JNLPBA-IOBES/train.tsv \
  $DATA_FOLDER/linnaeus-IOBES/train.tsv \
  --dev_file \
  $DATA_FOLDER/BC2GM-IOBES/devel.tsv \
  $DATA_FOLDER/BC4CHEMD-IOBES/devel.tsv \
  $DATA_FOLDER/NCBI-IOBES/devel.tsv \
  $DATA_FOLDER/JNLPBA-IOBES/devel.tsv \
  $DATA_FOLDER/linnaeus-IOBES/devel.tsv \
  --test_file \
  $DATA_FOLDER/BC2GM-IOBES/test.tsv \
  $DATA_FOLDER/BC4CHEMD-IOBES/test.tsv \
  $DATA_FOLDER/NCBI-IOBES/test.tsv \
  $DATA_FOLDER/JNLPBA-IOBES/test.tsv \
  $DATA_FOLDER/linnaeus-IOBES/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid nosig \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters $5 --epoch $5 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden $1 \
  --word_hidden $2 \
  --drop_out 0.5 \
  --pickle $SRC_FOLDER/pickle2 \
  --combine \
  --idea P14 \
  --pred_method M \
  --multi_mask $3 $4 \
  | tee $SRC_FOLDER/logs/$FOLDER/$EXEC_NAME.log


source deactivate
