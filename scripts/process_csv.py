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
import pandas as pd
from tqdm import tqdm
from coreLib.languages import vocab
from coreLib.processing import processLabels
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):
    csv         =   args.csv
    assert ".csv" in csv,"Not a csv file"
    # vocab lens
    glen        =   int(args.max_glen)
    ulen        =   int(args.max_ulen)
    # label processing
    vocab.unicodes.mlen=ulen
    vocab.graphemes.mlen=glen
    df=pd.read_csv(csv)
    df=processLabels(df,vocab)
    df.to_csv(csv,index=False)

    

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script with only label processing")
    parser.add_argument("csv", help="Path of the data.csv file")
    parser.add_argument("--max_glen",required=False,default=20,help=" the maximum length of grapheme data for modeling")
    parser.add_argument("--max_ulen",required=False,default=80,help=" the maximum length of unicode data for modeling")
    
    
    args = parser.parse_args()
    main(args)