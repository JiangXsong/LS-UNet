#!/bin/bash

# -- START IMPORTANT
speech=/home/song/LS-UNet/audio   #audio path
stage=1
# -- END

dumpdir=data  #directory ro put generated json file

# --START LS-UNet config
train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
sample_rate=16000
n_fft=128
frame_length=128

# Training config
use_cuda=1
id=0
epochs=100
half_lr=1
early_stop=0
max_norm=5

shuffle=1
batch_size=1
optimizer=adam
lr=2e-4
momentum=0
l2=0
# save
checkpoint=0
continue_from=""
print_freq=100
# --END LS-UNet config


tag="" # tag for managing experiments.

ngpu=1 # always 1

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  prepare_data.py --in_dir $speech --out_dir $dumpdir --sample_rate $sample_rate --n_fft $n_fft --frame_length $frame_length
fi


if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_epoch${epochs}_half${half_lr}_norm${max_norm}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_`basename $train_dir`
else
  expdir=exp/train_${tag}
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES="$id" \
    train.py \
    --train_dir $train_dir \
    --valid_dir $valid_dir \
    --sample_rate $sample_rate \
    --use_cuda $use_cuda \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --checkpoint $checkpoint \
    --continue_from "$continue_from" \
    --print_freq ${print_freq} \
    
fi