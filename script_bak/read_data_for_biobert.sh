#!/bin/bash

source activate


SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"

cd $SRC_FOLDER

python -u $SRC_FOLDER/read_data_for_biobert.py \
  --data_dir $SRC_FOLDER/corpus


conda deactivate