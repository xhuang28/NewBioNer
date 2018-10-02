#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

EXEC_NAME="EXP1"
SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="/auto/nlg-05/huan183/NewBioNer/corpus/train"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/$EXEC_NAME"


cd $SRC_FOLDER
mkdir $CHECKPOINT_FOLDER

python3 -u $SRC_FOLDER/ptrain.py \
  --checkpoint $CHECKPOINT_FOLDER \
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
  --dispatch N21 --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters 50 --epoch 500 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --char_hidden 200 --word_hidden 300 --drop_out 0.5 \
  --load_arg 0 \
  --load_check_point 0 \
  --load_opt \
  --pickle $SRC_FOLDER/pickle \
  | tee $SRC_FOLDER/logs/$EXEC_NAME.log


source deactivate
