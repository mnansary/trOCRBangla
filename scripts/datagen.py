#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
# TODO: add optional processing
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
import json
import pandas as pd 

from tqdm import tqdm
from ast import literal_eval

from coreLib.utils import *
from coreLib.processing import processData
from coreLib.store import createRecords
from coreLib.languages import vocab

tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_dir    =   args.data_dir
    iden        =   args.data_iden
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    seq_max_len =   int(args.seq_max_len)
    tf_size     =   int(args.rec_size)
    down_factor =   int(args.down_factor)
    img_dim=(img_height,img_width)
    # temporary save
    temp_dir=create_dir(data_dir,"temp")
    _=create_dir(temp_dir,"image")
    save_dir=create_dir(temp_dir,iden)
    config_json  =   "../config.json"
    
    # processing
    df=processData(data_dir,vocab,img_dim,seq_max_len)
    # storing
    createRecords(df,save_dir,img_dim,down_factor,tf_size=tf_size)
    config={"vocab":vocab,
        "pos_max":seq_max_len,
        "img_height":img_height,
        "img_width" :img_width,
        "tf_size":tf_size,
        "down_factor":down_factor}

    with open(config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("data_iden", help="identifier for the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    parser.add_argument("--seq_max_len",required=False,default=40,help=" the maximum length of data for modeling")
    parser.add_argument("--rec_size",required=False,default=10240,help=" the maximum length of data for storing in a tfrecord")
    parser.add_argument("--down_factor",required=False,default=32,help="the factor to downsample mask data")
    #parser.add_argument("--scene",required=False,type=str2bool,default=True,help ="wheather to use scene data:default=True")
    args = parser.parse_args()
    main(args)