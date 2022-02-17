#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
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
from coreLib.synthetic import createSyntheticData
from coreLib.languages import languages,vocab
from coreLib.processing import processLabels
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_dir    =   args.data_dir
    language    =   args.language
    data_type   =   args.data_type
    use_scene   =   args.scene
    exclude_punct=  args.exclude_punct
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    
    assert os.path.exists(os.path.join(data_dir,language)),"the specified language does not contain a valid graphems,numbers and fonts data"
    assert data_type in ["printed","handwritten"],"must be in ['printed','handwritten']"
    
    save_path   =   args.save_path
    pad_height  =   int(args.pad_height)
    num_samples =   int(args.num_samples)
    iden        =   args.iden

    # vocab lens
    glen        =   int(args.max_glen)
    ulen        =   int(args.max_ulen)

    img_dim=(img_height,img_width)

    if iden is None:
        iden=f"{language}_{data_type}"
    language=languages[language]
    # data creation
    df,csv=createSyntheticData(iden=iden,
                            save_dir=save_path,
                            data_dir=data_dir,
                            language=language,
                            img_dim=img_dim,
                            data_type=data_type,
                            num_samples=num_samples,
                            pad_height=pad_height,
                            create_scene_data=use_scene,
                            exclude_punct=exclude_punct)
    # label processing
    vocab.unicodes.mlen=ulen
    vocab.graphemes.mlen=glen
    df=processLabels(df,vocab)
    df.to_csv(csv,index=False)

    

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("language", help="the specific language to use")
    parser.add_argument("data_type", help="type of data to generate must be in ['printed','handwritten']")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--iden",required=False,default=None,help="identifier to identify the dataset")
    parser.add_argument("--pad_height",required=False,default=20,help ="pad height for each grapheme for alignment correction: default=20")
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when:default=100000")
    parser.add_argument("--scene",required=False,type=str2bool,default=True,help ="wheather to use scene data:default=True")
    parser.add_argument("--exclude_punct",required=False,type=str2bool,default=False,help ="wheather to avoid punctuations :default=False")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    parser.add_argument("--max_glen",required=False,default=20,help=" the maximum length of grapheme data for modeling")
    parser.add_argument("--max_ulen",required=False,default=80,help=" the maximum length of unicode data for modeling")
    
    
    args = parser.parse_args()
    main(args)