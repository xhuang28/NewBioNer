#!/bin/bash
source activate

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"

python -u $SRC_FOLDER/make_prediction.py \
  --checkpoint $SRC_FOLDER/trained_models/EXP2_CN21_C200_W300_MV0_EP58/N21_LAST_0.9710_0.9649_0.9773_58.model \
  --train_args $SRC_FOLDER/trained_models/EXP2_CN21_C200_W300_MV0_EP58/N21_LAST_0.9710_0.9649_0.9773_58.json_bak \
  --load_train_loader $SRC_FOLDER/dataloaders/crf2train_dataloader.p \
  --load_c_train_loader $SRC_FOLDER/dataloaders/combine/c_crf2train_dataloader.p \
  --save_file $SRC_FOLDER/dataloaders/EXP1

source deactivate