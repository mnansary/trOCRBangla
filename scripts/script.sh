#!/bin/sh



bw_ref="/home/apsisdev/ansary/DATASETS/RAW/bangla_writing/raw/raw/"
bh_ref="/home/apsisdev/ansary/DATASETS/RAW/BN-HTR/"
bs_ref="/home/apsisdev/ansary/DATASETS/RAW/BanglaC/README.txt"
iit_path="/home/apsisdev/Rezwan/cvit_iiit-indic/"
eng_hw_path="/home/apsisdev/ansary/DATASETS/RAW/eng_page/data/"
iam_path="/home/apsisdev/ansary/DATASETS/RAW/IAM_DATA/"

base_path="/home/apsisdev/ansary/DATASETS/APSIS/Recognition/"
#base_path="/home/ansary/WORK/Work/APSIS/datasets/Recognition/"

save_path=$base_path
src_dir="${base_path}source/"
batch_sample=1000000
#-----------------------------------------------------------------------------------------------
ds_path="${save_path}datasets/"
iit_bn_ref="${iit_path}bn/vocab.txt"

bw_ds="${ds_path}bw/"
bh_ds="${ds_path}bh/"
bs_ds="${ds_path}bs/"
iit_bn_ds="${ds_path}bn/"
en_ds="${ds_path}en/"
iam_ds="${ds_path}iam/"

enn_ds="${ds_path}enn/"
bnn_ds="${ds_path}bnn/"
bn_pr_ds="${ds_path}bangla_printed/"
en_pr_ds="${ds_path}english_printed/"

#-----------------------------------natrual---------------------------------------------
python datasets/bangla_writing.py $bw_ref $ds_path
python datasets/boise_state.py $bs_ref $ds_path
python datasets/bn_htr.py $bh_ref $ds_path
python datasets/iit_indic.py $iit_bn_ref $ds_path
python datasets/eng_hw.py $eng_hw_path $ds_path
python datasets/iam_eng.py $iam_path $ds_path
#python datagen.py $bw_ds 
#python datagen.py $bs_ds 
#python datagen.py $bh_ds 
#python datagen.py $iit_bn_ds 
#python datagen.py $bn_pr_ds
#python datagen.py $en_pr_ds
#python datagen.py $en_ds
#python datagen.py $iam_ds
#-----------------------------------natrual---------------------------------------------

#-----------------------------------synthetic------------------------------------------
#python synth.py $src_dir "bangla" "printed" $ds_path --iden "bnp" --num_samples 500000 --scene False --exclude_punct True
#python synth.py $src_dir "english" "printed" $ds_path --iden "enp" --num_samples 500000 --scene False --exclude_punct True
#python nums.py $src_dir "bangla" "handwritten" $ds_path --num_samples 100000 --iden "bnn" 
#python nums.py $src_dir "english" "handwritten" $ds_path --num_samples 100000 --iden "enn"
# python datagen.py $enn_ds
# python datagen.py $bnn_ds
#-----------------------------------synthetic------------------------------------------



echo succeeded