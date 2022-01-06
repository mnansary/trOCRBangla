# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

from coreLib.utils import LOG_INFO, create_dir
import argparse
import glob
import os
import numpy as np
import cv2
import pandas as pd
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET 
from xml.sax.saxutils import unescape as unescape_
#----------------------------------------------------------------------------------------------------
def unescape(s):
    return unescape_(s).replace('&quot;','"')
#----------------------------------------------------------------------------------------------------
def main(args):
    data_dir    =   args.data_path
    save_path   =   args.save_path
    save_path   =   create_dir(save_path,"iam")
    img_dir     =   create_dir(save_path,"images")
    csv         =   os.path.join(save_path,"data.csv")

    filepaths    =   []
    words        =   []
    fname        =   0


    for img_path in  tqdm(glob.glob(os.path.join(data_dir,"*.png"))):
        # image
        img=cv2.imread(img_path)
        # xml
        xml_path=img_path.replace(".png",".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for line in root.findall('./handwritten-part/line'):
            try:
                for word in line.findall('word'):
                    text=unescape(word.attrib['text'])
                    # component-wise
                    w_minX=99999999
                    w_maxX=-1
                    w_minY=99999999
                    w_maxY=-1
                    for cmp in word.findall('cmp'):
                        x = int(cmp.attrib['x'])
                        y = int(cmp.attrib['y'])
                        w = int(cmp.attrib['width'])
                        h = int(cmp.attrib['height'])
                        # update
                        w_maxX = max(w_maxX,x+w)
                        w_minX = min(w_minX,x)
                        w_maxY = max(w_maxY,y+h)
                        w_minY = min(w_minY,y)
                    wimg=img[w_minY:w_maxY+1,w_minX:w_maxX+1]
                    img_save_path=os.path.join(img_dir,f"{fname}.png")
                    cv2.imwrite(img_save_path,wimg)
                    filepaths.append(img_save_path)
                    words.append(text)
                    fname+=1
            except Exception as e:
                LOG_INFO(f"{text}-{os.path.basename(img_path)}")
                LOG_INFO(e,"red")
        
    # dictionary of lists 
    _dict = {'filepath': filepaths, 'word': words} 
    df = pd.DataFrame(_dict)

    # saving the dataframe
    df.to_csv(csv,index=False)
    

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("IAM English Handwritten word Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains .json and image files")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    args = parser.parse_args()
    main(args)
