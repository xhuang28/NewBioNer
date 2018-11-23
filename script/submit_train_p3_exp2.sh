#!/bin/bash

SRC_FOLDER="/auto/nlg-05/huan183/NewBioNer"
TRAIN_FOLDER="$SRC_FOLDER/corpus/train"
EVAL_FOLDER="$SRC_FOLDER/corpus/eval"

C1D="$TRAIN_FOLDER/BC2GM-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC4CHEMD-IOBES/devel.tsv \
    $TRAIN_FOLDER/BC5CDR-IOBES/devel.tsv \
    $TRAIN_FOLDER/NCBI-IOBES/devel.tsv \
    $TRAIN_FOLDER/JNLPBA-IOBES/devel.tsv \
    $TRAIN_FOLDER/linnaeus-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 15 False 1 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 68 False 1 "$C1D"


C2D="$EVAL_FOLDER/BioNLP11ID-IOBES/devel.tsv \
    $EVAL_FOLDER/BioNLP13CG-IOBES/devel.tsv \
    $EVAL_FOLDER/CRAFT-IOBES/devel.tsv"

#havn't run:
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 ?? False 2 "$C2D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 ?? False 2 "$C2D"



C3D="$EVAL_FOLDER/BioNLP11ID-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 32 False 3 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 257 False 3 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 8 False 4 "$C3D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 27 False 4 "$C3D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 2 False 5 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 5 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 40 False 6 "$C3D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 40 False 6 "$C3D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP257_RestartFalse_Comb3 N21_0.9943_0.9933_0.9954_214 M nosig 100 300 ?? False 3 "$C1D" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP40_RestartFalse_Comb6 N21_0.8920_0.8431_0.9469_40 M nosig 100 300 4 False 6 "$C3D" 2



C7D="$EVAL_FOLDER/BioNLP13CG-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 24 False 7 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 39 False 7 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 2 False 8 "$C7D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 8 "$C7D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 9 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 9 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 6 False 10 "$C7D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 42 False 10 "$C7D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP39_RestartFalse_Comb7 N21_0.9761_0.9709_0.9813_39 M nosig 100 300 ?? False 7 "$C1D" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP42_RestartFalse_Comb10 N21_0.8410_0.8516_0.8307_42 M nosig 100 300 90 False 10 "$C7D" 2



C11D="$EVAL_FOLDER/CRAFT-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 23 False 11 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 87 False 11 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 3 False 12 "$C11D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 19 False 12 "$C11D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 6 False 13 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 13 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 4 False 14 "$C11D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 67 False 14 "$C11D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP87_RestartFalse_Comb11 N21_LAST_0.9457_0.9119_0.9822_87 M nosig 100 300 ?? False 11 "$C1D" 2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP67_RestartFalse_Comb14 N21_LAST_0.7622_0.7901_0.7363_67 M nosig 100 300 73 False 14 "$C11D" 2



C15D="$EVAL_FOLDER/CELLFINDER-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 18 False 15 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 115 False 15 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 17 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 17 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP115_RestartFalse_Comb15 N21_0.9897_0.9850_0.9945_115 M nosig 100 300 ?? False 15 "$C1D" 2



C19D="$EVAL_FOLDER/CHEMPROT-IOBES/devel.tsv"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 17 False 19 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 97 False 19 "$C1D"

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P32 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 8 False 21 "$C1D"
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2.sh P33 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 1 False 21 "$C1D"


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP97_RestartFalse_Comb19 N21_0.9835_0.9730_0.9942_97 M nosig 100 300 ?? False 19 "$C1D" 2






# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb5_IT2 N21_0.9159_0.9618_0.8742_1 M nosig 100 300 ?? False 5 "$C1D" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP4_RestartFalse_Comb6_IT2 N21_0.8948_0.8482_0.9469_4 M nosig 100 300 ?? False 6 "$C3D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb9_IT2 N21_0.9436_0.9400_0.9473_1 M nosig 100 300 ?? False 9 "$C1D" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP90_RestartFalse_Comb10_IT2 N21_LAST_0.8487_0.8590_0.8387_90 M nosig 100 300 ?? False 10 "$C7D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb13_IT2 N21_0.8565_0.8433_0.8701_1 M nosig 100 300 ?? False 13 "$C1D" 3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP73_RestartFalse_Comb14_IT2 N21_LAST_0.7629_0.7951_0.7333_73 M nosig 100 300 ?? False 14 "$C11D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb17_IT2 N21_0.9451_0.9541_0.9364_1 M nosig 100 300 ?? False 17 "$C1D" 3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p3_exp2_iter.sh P33 P3/EXP2 P33_M_nosig_C100_W300_EP1_RestartFalse_Comb21_IT2 N21_0.9449_0.9361_0.9539_1 M nosig 100 300 ?? False 21 "$C1D" 3



