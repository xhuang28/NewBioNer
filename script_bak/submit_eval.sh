#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P10_M_nosig_C100_W300_MV0_EP51 N21_0.9652_0.9620_0.9683_51 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse N21_0.9618_0.9536_0.9702_9 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse N21_0.9618_0.9536_0.9702_9 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9762_0.9700_0.9826_29 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9762_0.9700_0.9826_29 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P22_P12_M_nosig_C100_W400_MV0.5_EP57_M_nosig_C100_W400_EP14_RestartFalse N21_0.9686_0.9593_0.9781_14 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P22_P12_M_nosig_C100_W400_MV0.5_EP57_M_nosig_C100_W400_EP14_RestartFalse N21_0.9686_0.9593_0.9781_14 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 U P23_P12_M_nosig_C100_W400_MV0.3_EP53_M_nosig_C100_W400_EP16_RestartFalse N21_0.9711_0.9612_0.9812_16 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W400_MV0.3_EP53_M_nosig_C100_W400_EP16_RestartFalse N21_0.9711_0.9612_0.9812_16 pickle2

# early stopping on single corpus
#BC5DR
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9592_0.9537_0.9649_36 pickle2
#NCBI
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9522_0.9486_0.9559_27 pickle2

#BC5DR
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9601_0.9518_0.9686_5 pickle2
#BC2GM, NCBI
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_0.9607_0.9508_0.9708_7 pickle2
#linnaeus
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP29_RestartFalse N21_BC5CDR-IOBES_0.9666_0.9643_0.9689_6 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P3 M P33_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP257_RestartFalse_Comb3 N21_0.9943_0.9933_0.9954_214 pickle2


# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP3_RestartFalse._linnaeus-IOBES N21_LAST_0.9909_0.9888_0.9930_3 linnaeus-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP9_RestartFalse._BC5CDR-IOBES N21_LAST_0.9885_0.9845_0.9926_9 BC5CDR-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP10_RestartFalse._NCBI-IOBES N21_LAST_0.9548_0.9312_0.9797_10 NCBI-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP14_RestartFalse._BC2GM-IOBES N21_LAST_0.9695_0.9635_0.9755_14 BC2GM-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP15_RestartFalse._BC4CHEMD-IOBES N21_LAST_0.9827_0.9735_0.9920_15 BC4CHEMD-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP20_RestartFalse._linnaeus-IOBES N21_LAST_0.9930_0.9861_1.0000_20 linnaeus-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP23_RestartFalse._JNLPBA-IOBES N21_LAST_0.9297_0.9186_0.9411_23 JNLPBA-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP3_RestartFalse._linnaeus-IOBES N21_LAST_0.9902_0.9861_0.9944_3 linnaeus-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP10_RestartFalse._JNLPBA-IOBES N21_LAST_0.9092_0.9008_0.9178_10 JNLPBA-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP10_RestartFalse._NCBI-IOBES N21_LAST_0.9467_0.9137_0.9822_10 NCBI-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP13_RestartFalse._BC2GM-IOBES N21_LAST_0.9761_0.9734_0.9788_13 BC2GM-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP15_RestartFalse._BC4CHEMD-IOBES N21_LAST_0.9843_0.9775_0.9913_15 BC4CHEMD-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP15_RestartFalse._BC5CDR-IOBES N21_LAST_0.9911_0.9868_0.9954_15 BC5CDR-IOBES

# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP100_RestartFalse._BC5CDR-IOBES N21_0.9987_0.9974_1.0000_81 NCBI-IOBES
# bash eval_single.sh P2 M P22_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP100_RestartFalse._NCBI-IOBES N21_0.9862_0.9740_0.9987_49 NCBI-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP100_RestartFalse._BC5CDR-IOBES N21_0.9986_0.9972_1.0000_88 NCBI-IOBES
# bash eval_single.sh P2 M P23_P12_M_nosig_C100_W300_MV0.2_EP49_M_nosig_C100_W300_EP100_RestartFalse._NCBI-IOBES N21_0.9868_0.9752_0.9987_55 NCBI-IOBES


# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.25_0.25_EP56 N21_LAST_0.9691_0.9637_0.9746_56 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.25_0.25_EP56 N21_LAST_0.9691_0.9637_0.9746_56 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.3_0.3_EP49 N21_LAST_0.9654_0.9593_0.9716_49 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.3_0.3_EP49 N21_LAST_0.9654_0.9593_0.9716_49 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.2_0.2_EP56 N21_LAST_0.9687_0.9620_0.9756_56 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.2_0.2_EP56 N21_LAST_0.9687_0.9620_0.9756_56 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.1_0.1_EP50 N21_LAST_0.9664_0.9595_0.9733_50 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.1_0.1_EP50 N21_LAST_0.9664_0.9595_0.9733_50 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.05_0.05_EP50 N21_LAST_0.9672_0.9612_0.9733_50 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.05_0.05_EP50 N21_LAST_0.9672_0.9612_0.9733_50 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.01_0.01_EP57 N21_LAST_0.9700_0.9633_0.9768_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.01_0.01_EP57 N21_LAST_0.9700_0.9633_0.9768_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C50_W300_MV0_0_EP66 N21_LAST_0.9712_0.9633_0.9792_66 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C50_W300_MV0_0_EP66 N21_LAST_0.9712_0.9633_0.9792_66 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C200_W300_MV0_0_EP62 N21_LAST_0.9728_0.9663_0.9795_62 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C200_W300_MV0_0_EP62 N21_LAST_0.9728_0.9663_0.9795_62 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0.001_0.001_EP57 N21_LAST_0.9690_0.9636_0.9746_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W300_MV0.001_0.001_EP57 N21_LAST_0.9690_0.9636_0.9746_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W400_MV0_0_EP57 N21_LAST_0.9712_0.9631_0.9795_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C100_W400_MV0_0_EP57 N21_LAST_0.9712_0.9631_0.9795_57 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C200_W400_MV0_0_EP54 N21_LAST_0.9721_0.9645_0.9797_54 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 U P14_M_nosig_C200_W400_MV0_0_EP54 N21_LAST_0.9721_0.9645_0.9797_54 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W300_MV0_0_EP84 N21_LAST_0.9765_0.9681_0.9849_84 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C100_W400_MV0_0_EP55 N21_LAST_0.9713_0.9666_0.9760_55 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C200_W300_MV0_0_EP60 N21_LAST_0.9698_0.9629_0.9767_60 pickle2
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_M_nosig_C200_W400_MV0_0_EP66 N21_LAST_0.9751_0.9665_0.9839_66 pickle2

# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_BC2GM-IOBES_Epoch5 N21_LAST_0.9682_0.9674_0.9690_89 pickle/BC2GM-IOBES
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_BC4CHEMD-IOBES_Epoch4 N21_LAST_0.9839_0.9764_0.9916_88 pickle/BC4CHEMD-IOBES
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_BC5CDR-IOBES_Epoch4 N21_LAST_0.9840_0.9790_0.9891_88 pickle/BC5CDR-IOBES
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_JNLPBA-IOBES_Epoch4 N21_LAST_0.9074_0.8988_0.9161_88 pickle/JNLPBA-IOBES
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_linnaeus-IOBES_Epoch3 N21_LAST_0.9937_0.9930_0.9944_87 pickle/linnaeus-IOBES
# sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M P14_NCBI-IOBES_Epoch4 N21_LAST_0.9697_0.9636_0.9759_88 pickle/NCBI-IOBES

sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_conll.sh P1 CONLL_C100_W300_MV0_0_EP76 N21_LAST_0.9972_0.9975_0.9970_76
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_conll.sh P1 CONLL_C100_W300_MV0_1_EP63 N21_LAST_0.0034_1.0000_0.0017_63
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval_conll.sh P1 CONLL_C100_W300_MV1_1_EP4  N21_LAST_0.5997_0.4503_0.8973_4


squeue -u huan183

