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
from coreLib.utils import create_dir
from coreLib.store import createRecords
from coreLib.languages import vocab

tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    csv         =   args.csv
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    max_ulen    =   int(args.max_ulen)
    max_glen    =   int(args.max_glen)
    tf_size     =   int(args.rec_size)
    data_dir    =   os.path.dirname(csv)
    save_path   =   create_dir(data_dir,"tfrecords")
    config_json =   os.path.join(data_dir,"config.json")   
    
    
    # storing
    createRecords(csv,save_path,tf_size=tf_size)
    config={"unicodes":vocab.unicodes.all,
            "graphemes":vocab.graphemes.all,
            "max_ulen":max_ulen,
            "max_glen":max_glen,
            "img_height":img_height,
            "img_width" :img_width,
            "tf_size":tf_size}

    with open(config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script")
    parser.add_argument("csv", help="Path of the data.csv file")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    parser.add_argument("--rec_size",required=False,default=5120,help=" the maximum length of data for storing in a tfrecord")
    parser.add_argument("--max_glen",required=False,default=20,help=" the maximum length of grapheme data for modeling")
    parser.add_argument("--max_ulen",required=False,default=80,help=" the maximum length of unicode data for modeling")
    
    args = parser.parse_args()
    main(args)