#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import math
import pandas as pd 
import tensorflow as tf
import numpy as np 
from ast import literal_eval
from tqdm.auto import tqdm
from .utils import *
tqdm.pandas()
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------

# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def update_feat(img_path,feat,data):
    feat_path=img_path.replace("images",feat)
    with(open(feat_path,'rb')) as fid:
        png_bytes=fid.read()
    # feature desc
    data[feat]=_bytes_feature(png_bytes)
    return data
    


def toTfrecord(df,rnum,rec_path):
    '''
        args:
            df      :   the dataframe that contains the information to store
            rnum    :   record number
            rec_path:   save_path
            mask_dim:   the dimension of the mask
    '''
    tfrecord_name=f'{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name)
    LOG_INFO(tfrecord_path) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in tqdm(range(len(df))):
            try:
                # base
                img_path=df.iloc[idx,0]
                # img
                with(open(img_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes)}
                data=update_feat(img_path,"image_masks",data)
                data=update_feat(img_path,"attention_masks",data)
                data=update_feat(img_path,"standards",data)
                
                # label
                data["ulabel"]=_int64_list_feature(df.iloc[idx,-2])
                data["glabel"]=_int64_list_feature(df.iloc[idx,-1]) 

                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            except Exception as e:
                continue

def createRecords(data,save_path,tf_size=10240):
    '''
        creates tf records:
        args:
            data        :   either the csv path or a dataframe cols=["filepath","word","datapath","label"]
            save_path   :   location to save tfrecords
    '''
    if type(data)==str:
        data=pd.read_csv(data)
        data.dropna(inplace=True)
        data["ulabel"]=data["ulabel"].progress_apply(lambda x: literal_eval(x))
        data["glabel"]=data["glabel"].progress_apply(lambda x: literal_eval(x))
        
    data.reset_index(drop=True,inplace=True)
    
    LOG_INFO(f"Creating TFRECORDS No folds:{save_path}")
    for idx in range(0,len(data),tf_size):
        df        =   data.iloc[idx:idx+tf_size] 
        df.reset_index(drop=True,inplace=True) 
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path)

    
    