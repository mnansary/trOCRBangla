#!/bin/sh



bw_ref="/backup/RAW/bangla_writing/raw/raw/"
bh_ref="/backup/RAW/BN-HTR/"
bs_ref="/backup/RAW/BanglaC/README.txt"
iit_bn_ref="/backup/RAW/iit.bn/vocab.txt"
eng_hw_path="/backup/RAW/eng_page/data/"
iam_path="/backup/RAW/IAM_DATA/"

src_dir="/home/apsisdev/ansary/DATASETS/APSIS/Recognition/source/"
ds_path="/backup/Recognition/datasets"

#src_dir="/home/ansary/WORK/Work/APSIS/datasets/Recognition/source/"

#-----------------------------------------------------------------------------------------------
#-----------------------------------natrual---------------------------------------------
# python datasets/bangla_writing.py $bw_ref $ds_path
# python datasets/boise_state.py $bs_ref $ds_path
# python datasets/bn_htr.py $bh_ref $ds_path
# python datasets/iit_indic.py $iit_bn_ref $ds_path
# python datasets/eng_hw.py $eng_hw_path $ds_path
# python datasets/iam_eng.py $iam_path $ds_path

#-----------------------------------natrual---------------------------------------------

#-----------------------------------synthetic------------------------------------------
batch_sample="1000000"
mid_sample="200000"
short_sample="50000"
#-----------------------------------synthetic------------------------------------------
# python synth.py $src_dir "bangla" "printed" $ds_path --iden "sbns" --num_samples $batch_sample --scene True --exclude_punct True
# python synth.py $src_dir "bangla" "printed" $ds_path --iden "sbnps" --num_samples $short_sample --scene True --exclude_punct False
# python synth.py $src_dir "english" "printed" $ds_path --iden "sens" --num_samples $mid_sample --scene True --exclude_punct True
# python synth.py $src_dir "english" "printed" $ds_path --iden "senps" --num_samples $short_sample --scene True --exclude_punct False

# python synth.py $src_dir "bangla" "printed" $ds_path --iden "sbn" --num_samples $batch_sample --scene False --exclude_punct True
# python synth.py $src_dir "bangla" "printed" $ds_path --iden "sbnp" --num_samples $short_sample --scene False --exclude_punct False
# python synth.py $src_dir "english" "printed" $ds_path --iden "sen" --num_samples $mid_sample --scene False --exclude_punct True
# python synth.py $src_dir "english" "printed" $ds_path --iden "senp" --num_samples $short_sample --scene False --exclude_punct False

# python synth.py $src_dir "bangla" "handwritten" $ds_path --iden "sbnh" --num_samples $batch_sample 

#-----------------------------------synthetic------------------------------------------



echo succeeded