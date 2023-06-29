#!/bin/bash -l

#SBATCH -J  abl-test           # --job-name=singularity
#SBATCH -o  abl-test.%j.out     # slurm output file (%j:expands to %jobId)
#SBATCH -p  A100-80GB             # queue or partiton name ; sinfo  output
#SBATCH -t  72:00:00          # 작업시간(hh:mm:ss) 1.5 hours 설정
#SBATCH -N  1                 # --nodes=1 (고정)
#SBATCH -n  1                 # --tasks-per-node=1  노드당 할당한 작업수 (고정)
#SBATCH --gres=gpu:1          # gpu Num Devices  가급적이면  1,2,4.6,8  2배수로 합니다.

conda activate hug4.29

EPOCHS=50
BATCH_SIZE=4
LR=3e-5
WARMUP=1000

OUTPUT=output-musique
TAG=$EPOCHS.$BATCH_SIZE.$LR.$WARMUP
DATA_DIR=/home/seonjeongh/data/musique/experiment

python run.py --do_test \
              --epochs $EPOCHS \
              --batch_size $BATCH_SIZE \
              --learning_rate $LR \
              --warmup_steps $WARMUP \
              --output $OUTPUT \
              --exp_tag $TAG \
              --checkpoint $OUTPUT/$TAG/12_49.15313323385143.pth

