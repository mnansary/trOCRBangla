# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

from coreLib.utils import create_dir
import argparse
import glob
import os
import numpy as np
import cv2
import pandas as pd
import json
from tqdm import tqdm
#----------------------------------------------------------------------------------------------------
def main(args):
    data_dir    =   args.data_path
    save_path   =   args.save_path
    save_path   =   create_dir(save_path,"en")
    img_dir     =   create_dir(save_path,"images")
    csv         =   os.path.join(save_path,"data.csv")


    path        =   glob.glob(os.path.join(data_dir,"*.jpg"))
    filepath    =   []
    word        =   []
    fname       =   0
    
    for img_file in tqdm(path):
        img_file_name = os.path.basename(img_file)
        file_name = img_file_name.split('.')[0]

        img = cv2.imread(img_file)
        f = open( os.path.join(data_dir,f"{file_name}.json"))
        data = json.load(f)

        for i in range(len(data)):
            poly = data[i]['polygon']
            x0 = poly['x0']
            y0 = poly['y0']
            x1 = poly['x1']
            y1 = poly['y1']
            x2 = poly['x2']
            y2 = poly['y2']
            x3 = poly['x3']
            y3 = poly['y3']

            text = data[i]['text']
            pts = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])

            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            croped = img[y:y+h, x:x+w].copy()
            img_save_path=os.path.join(img_dir,f"{fname}.png")

            cv2.imwrite(img_save_path, croped)
            
            filepath.append(img_save_path)
            word.append(text)
            fname+=1
    # dictionary of lists 
    _dict = {'filepath': filepath, 'word': word} 
    df = pd.DataFrame(_dict)

    # saving the dataframe
    df.to_csv(csv,index=False)
        
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("English Handwritten word Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains .json and image files")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    args = parser.parse_args()
    main(args)
