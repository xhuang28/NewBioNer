#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

CHECKPOINT_PATH="${1}"
CHECKPOINT_NAME="${2}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
LOAD_CHECKPOINT="$SRC_FOLDER/checkpoints/$CHECKPOINT_PATH"
IDX_ITER="IT${6}"

cd $SRC_FOLDER
mkdir data_loaders/$CHECKPOINT_PATH

python3 -u $SRC_FOLDER/upredict.py \
  --load_check_point $LOAD_CHECKPOINT/$CHECKPOINT_NAME.model \
  --load_arg $LOAD_CHECKPOINT/$CHECKPOINT_NAME.json \
  --data_loader $SRC_FOLDER/data_loaders/$CHECKPOINT_PATH$IDX_ITER \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file \
  $DATA_FOLDER/BC2GM-IOBES/train.tsv \
  $DATA_FOLDER/BC4CHEMD-IOBES/train.tsv \
  $DATA_FOLDER/BC5CDR-IOBES/train.tsv \
  $DATA_FOLDER/NCBI-IOBES/train.tsv \
  $DATA_FOLDER/JNLPBA-IOBES/train.tsv \
  $DATA_FOLDER/linnaeus-IOBES/train.tsv \
  --dev_file \
  $DATA_FOLDER/BC2GM-IOBES/devel.tsv \
  $DATA_FOLDER/BC4CHEMD-IOBES/devel.tsv \
  $DATA_FOLDER/BC5CDR-IOBES/devel.tsv \
  $DATA_FOLDER/NCBI-IOBES/devel.tsv \
  $DATA_FOLDER/JNLPBA-IOBES/devel.tsv \
  $DATA_FOLDER/linnaeus-IOBES/devel.tsv \
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
  --pickle $SRC_FOLDER/pickle2 \
  --combine

source deactivate
