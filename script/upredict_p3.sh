#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="${1}" # Fix me!

ORIG_PATH="${2}"
CHECKPOINT_PATH="${3}"
CHECKPOINT_NAME="${4}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints/$ORIG_PATH/$CHECKPOINT_PATH"

cd $SRC_FOLDER
mkdir data_loaders/$FOLDER/$CHECKPOINT_PATH
mkdir $SRC_FOLDER/pickle3/$1/$8

python3 -u $SRC_FOLDER/upredict_p3_args.py \
  --load_check_point $LOAD_CHECKPOINT/$CHECKPOINT_NAME.model \
  --load_arg $LOAD_CHECKPOINT/$CHECKPOINT_NAME.json \
  --data_loader $SRC_FOLDER/data_loaders/$FOLDER/$CHECKPOINT_PATH \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file $9 \
  --test_as_train ${10} \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid $5 \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch 500 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden $6 \
  --word_hidden $7 \
  --drop_out 0.5 \
  --idx_combination $8 \
  --pickle $SRC_FOLDER/pickle3/$1/$8

source deactivate
