#!/bin/bash

source /usr/usc/cuDNN/v7.0.5-cuda9.0/setup.sh
source /usr/usc/cuda/9.0/setup.sh

source activate bert

BERT_DIR="/auto/nlg-05/huan183/biobert"
BERT_BASE_DIR="$BERT_DIR/pubmed_pmc_470k"
DATA_DIR="/auto/nlg-05/huan183/NewBioNer/corpus"

cd $BERT_DIR

python extract_features.py \
  --input_file=$DATA_DIR/${1}.txt \
  --output_file=$DATA_DIR/${1}.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/biobert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=500 \
  --batch_size=8


conda deactivate