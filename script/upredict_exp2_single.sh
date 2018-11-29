#!/bin/bash

source activate bioner

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

CHECKPOINT_PATH="${1}"
CHECKPOINT_NAME="${2}"

SRC_FOLDER="/media/storage_e/npeng/bioner/xiao/github/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints/$CHECKPOINT_PATH"

cd $SRC_FOLDER
mkdir data_loaders/$CHECKPOINT_PATH._$6
mkdir pickle2/$6

python3 -u $SRC_FOLDER/upredict.py \
  --load_check_point $LOAD_CHECKPOINT/$CHECKPOINT_NAME.model \
  --load_arg $LOAD_CHECKPOINT/$CHECKPOINT_NAME.json \
  --data_loader $SRC_FOLDER/data_loaders/$CHECKPOINT_PATH._$6 \
  --emb_file /home/npeng/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file $DATA_FOLDER/$6/train.tsv \
  --dev_file $DATA_FOLDER/$6/devel.tsv \
  --test_file \
  $DATA_FOLDER/BC2GM-IOBES/test.tsv \
  $DATA_FOLDER/BC4CHEMD-IOBES/test.tsv \
  $DATA_FOLDER/BC5CDR-IOBES/test.tsv \
  $DATA_FOLDER/NCBI-IOBES/test.tsv \
  $DATA_FOLDER/JNLPBA-IOBES/test.tsv \
  $DATA_FOLDER/linnaeus-IOBES/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --sigmoid $3 \
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch 500 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden $4 \
  --word_hidden $5 \
  --drop_out 0.5 \
  --pickle $SRC_FOLDER/pickle2/$6 \
  --combine

source deactivate
