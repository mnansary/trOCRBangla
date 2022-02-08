# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import pandas as pd 
import cv2
import math
from tqdm import tqdm
from .utils import *
tqdm.pandas()
#--------------------
# helpers
#--------------------
#--------------------
# helpers
#--------------------
not_found=[]

def reset(df):
    # sort df
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True) 
    return df

def cvt_str(x):
    try:
        x=str(x)
        x=x.strip()
        x=x.replace(" ","")
        return x
    except Exception as e:
        return None


def encode_label(x,vocab,max_len):
    '''
        encodes a label
    '''
    global not_found
    label=[]
    x=["start"]+x+["end"]
    if len(x)>max_len:
        return None
    else:    
        for ch in x:
            try:
                label.append(vocab.index(ch))
            except Exception as e:
                if ch not in not_found:not_found.append(ch)
        pad=[vocab.index("pad") for _ in range(max_len-len(x))]
        return label+pad

def padWordImage(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")    
#---------------------------------------------------------------
def correctPadding(img,dim,ptype="central",pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#---------------------------------------------------------------
def processLabels(df,vocab):
    '''
        processLabels:
        * divides: word to - unicodes,components,graphemes
    '''
    GP=GraphemeParser(language=None)
    # process text
    df.word=df.word.progress_apply(lambda x:cvt_str(x))
    df=reset(df)
    # unicodes
    df["u"]=df.word.progress_apply(lambda x:[i for i in x])
    df=reset(df)
    df["ulabel"]=df["u"].progress_apply(lambda x:encode_label(x,vocab.unicodes.all,vocab.unicodes.mlen))
    df=reset(df)
    
    # components
    df["c"]=df.word.progress_apply(lambda x:GP.process(x,return_graphemes=False))
    df=reset(df)
    df["clabel"]=df["c"].progress_apply(lambda x:encode_label(x,vocab.components.all,vocab.components.mlen))
    df=reset(df)
    
    # graphemes
    df["g"]=df.word.progress_apply(lambda x:GP.process(x,return_graphemes=True))
    df=reset(df)
    df["glabel"]=df["g"].progress_apply(lambda x:encode_label(x,vocab.graphemes.all,vocab.graphemes.mlen))
    df=reset(df)
    return df 

# #---------------------------------------------------------------
# def processImages(df,save_dir,img_dim,ptype="left"):
#     '''
#         process a specific dataframe with filename,word,graphemes and mode
#         args:
#             df      :   the dataframe to process
#             save_dir:   path to save temp data
#             img_dim :   tuple of (img_height,img_width)  
#             ptype   :   type of padding to use
#     '''
#     datapaths=[]
#     img_dir=os.path.join(save_dir,"image")
#     mask_dir=os.path.join(save_dir,"mask")

#     for idx in tqdm(range(len(df))):
#         try:
#             # mask
#             mask=np.zeros(img_dim)
#             # path
#             img_path    =   df.iloc[idx,0]
#             # filename
#             file_name   =   os.path.basename(img_path)
#             # read image
#             img=cv2.imread(img_path)
#             # correct padding
#             img,imask=correctPadding(img,img_dim,ptype=ptype)
#             # mask
#             mask[:,imask:]=1
#             datapath=os.path.join(img_dir,file_name)
#             cv2.imwrite(datapath,img)
#             cv2.imwrite(os.path.join(mask_dir,file_name),mask)
#             datapaths.append(datapath)
            
#         except Exception as e:
#             datapaths.append(None)
#             LOG_INFO(e)
#     df["datapath"]=datapaths
#     return df

# #------------------------------------------------
# def processData(data_dir,vocab,img_dim,max_len):
#     '''
#         processes the dataset
#         args:
#             data_dir    :   the directory that holds data.csv and images folder
#             vocab       :   language class
#             img_dim     :   tuple of (img_height,img_width) 
#             max_len     :   model max_len
#     '''
#     csv=os.path.join(data_dir,"data.csv")
#     save_dir=os.path.join(data_dir,"temp")
#     # processing
#     df=pd.read_csv(csv)
#     df.reset_index(drop=True,inplace=True)
#     # images
#     df=processImages(df,save_dir,img_dim)
#     # labels
#     df=processLabels(df,vocab,max_len)
#     # save data
#     cols=["filepath","word","datapath","label"]
#     df=df[cols]
#     df.dropna(inplace=True)
#     df.reset_index(drop=True,inplace=True)
#     df.to_csv(csv,index=False)
#     LOG_INFO(f"Not Found:{not_found}")
#     return df