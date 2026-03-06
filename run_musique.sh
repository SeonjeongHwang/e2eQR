#!/bin/bash -l

EPOCHS=50
BATCH_SIZE=4
LR=3e-5
WARMUP=1000

OUTPUT=output_musique
TAG=$EPOCHS.$BATCH_SIZE.$LR.$WARMUP

mkdir $OUTPUT

python run_e2eqr.py \
              --do_train \
              --do_test \
              --train_data_file musique_train_bridge.json \
              --valid_data_file musique_dev_bridge.json \
              --test_data_file musique_test_bridge.json \
              --epochs $EPOCHS \
              --batch_size $BATCH_SIZE \
              --learning_rate $LR \
              --warmup_steps $WARMUP \
              --output $OUTPUT \
              --exp_tag $TAG

