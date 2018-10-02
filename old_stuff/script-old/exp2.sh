#!/bin/bash

source activate py36

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

EXEC_NAME="EXP2_N21_C${1}_W${2}_MV${3}"
SRC_FOLDER="/home/nlg-05/lidong/clean_base/MT_NER"
DATA_FOLDER="/home/nlg-05/lidong/file4bioner/EXP2/corpus/train"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoint/$EXEC_NAME"


cd $SRC_FOLDER
mkdir $CHECKPOINT_FOLDER

python3 $SRC_FOLDER/ptrain.py \
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
  --dispatch N21 --corpus_mask_value $3 \
  --batch_size 10 \
  --least_iters $6 --epoch $7 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu $8 \
  --char_hidden $1 --word_hidden $2 --drop_out 0.5 \
  --load_arg $4 \
  --load_check_point $5 \
  --load_opt \
  --pickle $SRC_FOLDER/pickle/EXP2/$3 \
  | tee $SRC_FOLDER/log/EXP2/$EXEC_NAME.log
  
  
source deactivate py36
