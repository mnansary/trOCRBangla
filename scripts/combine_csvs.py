#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import os
from glob import glob 
from tqdm.auto import tqdm
import pandas as pd
#--------------------
# main
#--------------------
def main(args):
    data_dir    =   args.data_dir
    csvs=[csv for csv in glob(os.path.join(data_dir,"*/*.csv"))]
    dfs=[]
    for csv in tqdm(csvs):
        df=pd.read_csv(csv)
        df=df[["filepath","word"]]
        dfs.append(df)
    df=pd.concat(dfs,ignore_index=True)
    df=df.sample(frac=1)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    csv         =   os.path.join(data_dir,"data.csv")
    df.to_csv(csv,index=False)

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script: dataset merging")
    parser.add_argument("data_dir", help="Path of the folder that contains multiple datasets")
    args = parser.parse_args()
    main(args)