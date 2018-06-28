#!/bin/bash
source activate py36

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

EXEC_NAME="Combine_Silver_N2N_C${1}_W${2}_DP${3}"
SRC_FOLDER="/home/nlg-05/lidong/clean_base/MT_NER"
DATA_FOLDER="/home/nlg-05/lidong/clean_base/MT_NER/corpus"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoint/$EXEC_NAME"

cd $SRC_FOLDER
mkdir $CHECKPOINT_FOLDER

python3 $SRC_FOLDER/ptrain.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file \
  $DATA_FOLDER/Silver-IOBES/train.tsv \
  --dev_file \
  $DATA_FOLDER/Silver-IOBES/devel.tsv \
  --test_file \
  $DATA_FOLDER/CELLFINDER-IOBES/test.tsv \
  $DATA_FOLDER/CHEMPROT-IOBES/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --dispatch N2N --corpus_mask_value 1.0 \
  --least_iters 43 --epoch 43 --patience 30 --stop_on_single \
  --lr 0.01 \
  --gpu 1 \
  --char_hidden $1 --word_hidden $2 --drop_out $3 \
  --load_arg $4 \
  --load_check_point $5 \
  --load_opt \
  --combine \
  --pickle $SRC_FOLDER/pickle/csver  \
  | tee $SRC_FOLDER/log/$EXEC_NAME.log
source deactivate py36
