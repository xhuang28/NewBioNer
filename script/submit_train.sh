#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp1.sh 100 300 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp1.sh 200 300 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp1.sh 100 300 1 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp1.sh 200 300 1 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp1.sh 100 300 0 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp1.sh 200 300 0 0

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_exp1_MTL.sh 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_exp1_MTL.sh 200 300

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/BC5CDR-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/BC5CDR-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/BioNLP11ID-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/BioNLP11ID-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/BioNLP13CG-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/BioNLP13CG-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/CRAFT-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh eval/CRAFT-IOBES 200 300

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/BC2GM-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/BC2GM-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/BC4CHEMD-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/BC4CHEMD-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/JNLPBA-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/JNLPBA-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/linnaeus-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/linnaeus-IOBES 200 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/NCBI-IOBES 100 300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh train/NCBI-IOBES 200 300


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --nodelist=hpc4326 ./train_p14_exp2.sh 100 300 0 1 200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp2.sh 200 300 0 1 200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --nodelist=hpc4328 ./train_p14_exp2.sh 100 300 1 1 200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp2.sh 200 300 1 1 200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --nodelist=hpc4329 ./train_p14_exp2.sh 100 300 0 0 200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p14_exp2.sh 200 300 0 0 200

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_exp2_MTL.sh 100 300 425
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_exp2_MTL.sh 200 300 382

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/BC5CDR-IOBES 100 300 58
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/BC5CDR-IOBES 200 300 50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/BioNLP11ID-IOBES 100 300 20
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/BioNLP11ID-IOBES 200 300 49
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/BioNLP13CG-IOBES 100 300 51
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/BioNLP13CG-IOBES 200 300 82
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/CRAFT-IOBES 100 300 59
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh eval/CRAFT-IOBES 200 300 62

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/BC2GM-IOBES 100 300 44
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/BC2GM-IOBES 200 300 78
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/BC4CHEMD-IOBES 100 300 30
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/BC4CHEMD-IOBES 200 300 63
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/JNLPBA-IOBES 100 300 46
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/JNLPBA-IOBES 200 300 44
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/linnaeus-IOBES 100 300 17
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/linnaeus-IOBES 200 300 23
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/NCBI-IOBES 100 300 22
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh train/NCBI-IOBES 200 300 139


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 100 300 BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 200 300 BC5CDR-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 100 300 BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 200 300 BioNLP11ID-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 100 300 BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 200 300 BioNLP13CG-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 100 300 CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp1.sh 200 300 CRAFT-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp1.sh 100 300 CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 100 300 BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425 137
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 200 300 BC5CDR-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382 117

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 100 300 BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425 109
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 200 300 BioNLP11ID-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382 49

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 100 300 BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425 45
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 200 300 BioNLP13CG-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382 43

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 100 300 CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425 80
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM_exp2.sh 200 300 CRAFT-IOBES MTL_C200_W300/N2N_BC2GM-IOBES_LAST_0.9296_0.9275_0.9317_382 147

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0 53
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1 37
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1 60

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0 20
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1 23
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1 20

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0 33
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1 46
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1 46

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0 28
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1 40
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_exp2.sh 100 300 CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1 67



# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BC5CDR-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425      1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP11ID-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh BioNLP13CG-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425  1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh CRAFT-IOBES MTL_C100_W300/N2N_BC4CHEMD-IOBES_LAST_0.9706_0.9616_0.9797_425       1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0        1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1      1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BC5CDR-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1         1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP11ID-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0    1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1  1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh BioNLP13CG-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1     1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_0_EP200/N21_JNLPBA-IOBES_0.9012_0.8876_0.9153_63 0 0         1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV0_1_EP200/N21_BC4CHEMD-IOBES_0.9746_0.9688_0.9805_42 0 1       1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh CRAFT-IOBES 100_W300_MV1_1_EP200/N21_BC2GM-IOBES_0.8924_0.8322_0.9621_86 1 1          1000

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      2000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES  2000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES  2000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       600
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       1000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/CRAFT-IOBES       2000




# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_MTM.sh conll2003_converted CONLL_MTL_C100_W300/N2N_LOC_LAST_0.9900_0.9865_0.9935_300      400

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_0_EP76/N21_LAST_0.9972_0.9975_0.9970_76 0 0        400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV0_1_EP63/N21_LAST_0.0034_1.0000_0.0017_63 0 1        400
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         100
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         150
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         200
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         250
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         300
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         350
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune.sh conll2003_converted CONLL_C100_W300_MV1_1_EP4/N21_LAST_0.5997_0.4503_0.8973_4   1 1         400


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh conll2003_converted      2000
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh conll2003_converted      all
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BC5CDR-IOBES      all
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP11ID-IOBES      all
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./fine_tune_STM.sh eval/BioNLP13CG-IOBES      all


squeue -u huan183

