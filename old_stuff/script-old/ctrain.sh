#!/bin/bash

source activate mt-ner

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

# Parameters
# $1 - mask
# $2 - ending epoch

EXEC_NAME="cmask$1"
SRC_FOLDER="/home/nlg-05/tianyume/mt-ner"
CHECKPOINT_FOLDER="/home/nlg-05/tianyume/checkpoints/mt-ner/mask-exp/$EXEC_NAME"
DATA_FOLDER="/home/nlg-05/lidong/file4bioner/EXP2/corpus/train"
LOGS_FOLDER="/home/nlg-05/tianyume/logs/mt-ner/mask-exp"

cd $SRC_FOLDER
mkdir $CHECKPOINT_FOLDER

python3 $SRC_FOLDER/train.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --emb_file /home/nlg-05/tianyume/wikipedia-pubmed-and-PMC-w2v.txt \
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
  --dispatch N21 \
  --corpus_mask_value $1 \
  --batch_size 10 \
  --least_iters $2 \
  --epoch $2 \
  --patience 30 --stop_on_single \
  --lr 0.01 \
  --char_hidden 200 \
  --word_hidden 300 \
  --drop_out 0.5 \
  --combine \
  | tee $LOGS_FOLDER/$EXEC_NAME.log

source deactivate
