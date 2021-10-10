#!/bin/bash

# Exit on error
set -e
set -o pipefail


# General
# dataset dir
data_dir="../../LibriTTS/wav22k"
# dir to put the intermediate wavs
preprocess_dir="./res"
# dir that puts mels you want to inference
test_dir="../../mels/gt_trans"
# output dir, including ckpt, log, and inferenced mels
output_dir="/work/b07502172/universal_adaptor/results_libritts_random"
# python path you want to use
python_path=python3
# Controls from which stage to start
stage=2  
# Controls the directory name associated to the experiment
exp_name="unet_libritts_cutconfig" 
# gpu id
id=0


# Training
random_config=100
batch_size=32
segment_length=200
n_workers=4
valid_steps=3600



if [[ $stage -le  0 ]]; then
    echo "Stage 0: Converting files in datasets into intermediate wav files"
    CUDA_VISIBLE_DEVICES=$id $python_path rand_prep.py \
        --data $data_dir \
        --n_cfg $random_config \
        --n_workers $n_workers \
        --outdir $preprocess_dir
fi


if [[ $stage -le 1 ]]; then
    echo "Stage 1: Training"
    CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--data $data_dir \
        --preprocess_dir $preprocess_dir \
        --out_dir $output_dir \
        --exp_name $exp_name \
        --n_workers $n_workers \
        --batch_size $batch_size \
        --segment_length $segment_length \
        --valid_steps $valid_steps
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path inference.py \
		--data_dir $test_dir \
		--out_dir $output_dir \
		--exp_name $exp_name \
        --num_workers $n_workers
fi
