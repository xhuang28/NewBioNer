#!/bin/bash
source activate base

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8


EXP_NAME="EXP1"
EXEC_NAME="EXP1.$1"
SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
CHECKPOINT_FOLDER="$SRC_FOLDER/checkpoints/c_$EXEC_NAME"
DATA_FOLDER="$SRC_FOLDER/corpus/train"
LOGS_FOLDER="$SRC_FOLDER/logs"
DATA_LOADER_FOLDER="$SRC_FOLDER/dataloaders"
#LOAD_CHECKPOINT="/auto/nlg-05/huan183/NewBioNer/trained_models/EXP2_CN21_C200_W300_MV0_EP58/N21_LAST_0.9710_0.9649_0.9773_58"


cd $SRC_FOLDER
mkdir $CHECKPOINT_FOLDER

python3 $SRC_FOLDER/train.py \
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
  --dispatch N21 \
  --corpus_mask_value 0 \
  --batch_size 10 \
  --least_iters $2 \
  --epoch $2 \
  --patience 30 \
  --stop_on_single \
  --lr 0.01 \
  --char_hidden 200 \
  --word_hidden 300 \
  --drop_out 0.5 \
  --load_arg $LOAD_CHECKPOINT.json \
  --load_check_point $LOAD_CHECKPOINT.model \
  --combine \
  --train_loader $DATA_LOADER_FOLDER/$EXP_NAME/$EXEC_NAME/c_new_crf2train_dataloader.p \
  --dev_loader $DATA_LOADER_FOLDER/crf2dev_dataloader.p \
  --dev_loader2 $DATA_LOADER_FOLDER/dev_dataset_loader.p \
  --test_loader $DATA_LOADER_FOLDER/test_dataset_loader.p \
  | tee $LOGS_FOLDER/c_$EXEC_NAME.log

  source deactivate
