#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="P1/Fine_tune_P14" # Fix me!

EXEC_NAME="FT_P14_${1}_MV${3}_${4}_N${5}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/$FOLDER/$EXEC_NAME"
LOAD_CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/P1/EXP2"


cd $SRC_FOLDER
mkdir -p $CHECKPOINT_FOLDER
mkdir -p $SRC_FOLDER/pickle/Fine_tune_P14/${1}_$5
mkdir -p $SRC_FOLDER/logs/$FOLDER

python3 -u $SRC_FOLDER/train_p1.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --load_check_point $LOAD_CHECKPOINT_FOLDER/$2.model \
  --load_arg $LOAD_CHECKPOINT_FOLDER/$2.json \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file \
  $SRC_FOLDER/corpus/eval/$1/train_${5}.tsv \
  --dev_file \
  $SRC_FOLDER/corpus/eval/$1/devel.tsv \
  --test_file \
  $SRC_FOLDER/corpus/eval/$1/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid nosig \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch 500 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden 100 \
  --word_hidden 300 \
  --drop_out 0.5 \
  --pickle $SRC_FOLDER/pickle/Fine_tune_P14/${1}_$5 \
  --idea P14 \
  --pred_method M \
  --multi_mask $3 $4 \
  | tee $SRC_FOLDER/logs/$FOLDER/$EXEC_NAME.log


conda deactivate
