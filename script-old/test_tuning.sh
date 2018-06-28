#!/bin/bash
source activate py36

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

EXEC_NAME="N21_MM_C${1}_W${2}_DP${3}_GRP${4}"
SRC_FOLDER="/home/nlg-05/lidong/clean_base/MT_NER"
DATA_FOLDER="/home/nlg-05/lidong/clean_base/MT_NER/corpus"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoint/$EXEC_NAME"

cd $SRC_FOLDER
mkdir $CHECKPOINT_FOLDER

python3 $SRC_FOLDER/train.py \
  --checkpoint $CHECKPOINT_FOLDER \
  --train_file \
  $DATA_FOLDER/BC5CDR-chem-IOBES/train.tsv \
  $DATA_FOLDER/BC5CDR-disease-IOBES/train.tsv \
  --dev_file \
  $DATA_FOLDER/BC5CDR-chem-IOBES/devel.tsv \
  $DATA_FOLDER/BC5CDR-disease-IOBES/devel.tsv \
  --test_file \
  $DATA_FOLDER/BC5CDR-chem-IOBES/test.tsv \
  $DATA_FOLDER/BC5CDR-disease-IOBES/test.tsv \
  --word_dim 200 --char_dim 30 --caseless --fine_tune --shrink_embedding \
  --dispatch N21 --corpus_mask_value 0.0 \
  --least_iters 50 --epoch 1 --patience 15 --stop_on_single \
  --lr 0.01 \
  --gpu 0 \
  --max_margin --change_gold \
  --char_hidden $1 --word_hidden $2 --drop_out $3 --change_prob $4 \
  --load_check_point $6 \
  --load_arg $5 \
  --load_opt \
  | tee $SRC_FOLDER/log/tuning/$EXEC_NAME.log
source deactivate py36
