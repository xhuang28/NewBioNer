#!/bin/bash

source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

FOLDER="P1/Fine_tune" # Fix me!

# 1: idea; 2: char hidden size; 3: word hidden size; 4: mask value (please set 0 if idea!="Li");
EXEC_NAME="Analyze_FT_conll2003_converted_MV${2}_${3}_N${4}"

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
DATA_FOLDER="$SRC_FOLDER/corpus/eval"
LOAD_CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/P1/Fine_tune"


cd $SRC_FOLDER

python3 -u $SRC_FOLDER/analyze.py \
  --load_check_point $LOAD_CHECKPOINT_FOLDER/$1.model \
  --load_arg $LOAD_CHECKPOINT_FOLDER/$1.json \
  --emb_file /home/nlg-05/lidong/clean_base/MT_NER/external/embedding/wikipedia-pubmed-and-PMC-w2v.txt \
  --train_file \
  $SRC_FOLDER/corpus/conll2003_converted/train_${4}.tsv \
  --dev_file \
  $SRC_FOLDER/corpus/conll2003_converted/devel.tsv \
  --test_file \
  $SRC_FOLDER/corpus/conll2003_converted/test.tsv \
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
  --pickle $SRC_FOLDER/pickle/Fine_tune/conll2003_converted_$4 \
  --idea P10 \
  --pred_method M \
  --multi_mask $2 $3 \
  | tee $SRC_FOLDER/logs/$FOLDER/$EXEC_NAME.log


conda deactivate