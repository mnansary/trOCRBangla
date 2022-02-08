# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import cv2
import numpy as np
import random
import pandas as pd 
from tqdm import tqdm
from .utils import *
from .dataset import DataSet
from .languages import languages
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 
tqdm.pandas()
import matplotlib.pyplot as plt
import math
from .processing import correctPadding
noise=Modifier()
#--------------------
# helpers
#--------------------
def createImgFromComps(df,comps,pad):
    '''
        creates a synthetic image from given comps
        args:
            df      :       dataframe holding : "filename","label","img_path"
            comps   :       list of graphemes
            pad     :       pad class:
                                no_pad_dim
                                single_pad_dim
                                double_pad_dim
                                top
                                bot
        returns:
            non-pad-corrected raw binary image
    '''
    # get img_paths
    img_paths=[]
    for idx,comp in enumerate(comps):
        cdf=df.loc[df.label==comp]
        cdf=cdf.sample(frac=1)
        cdf.reset_index(drop=True,inplace=True)
        img_paths.append(cdf.iloc[0,2])
    
    # get images
    imgs=[cv2.imread(img_path,0) for img_path in img_paths]
    # alignment of component
    ## flags
    tp=False
    bp=False
    comp_heights=["" for _ in comps]
    for idx,comp in enumerate(comps):
        if any(te.strip() in comp for te in pad.top):
            comp_heights[idx]+="t"
            tp=True
        if any(be in comp for be in pad.bot):
            comp_heights[idx]+="b"
            bp=True

    # image construction based on height flags
    '''
    class pad:
        no_pad_dim      =(comp_dim,comp_dim)
        single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
        double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
        top             =top_exts
        bot             =bot_exts
        height          =pad_height  
    '''
    cimgs=[]
    for img,hf in zip(imgs,comp_heights):
        if hf=="":
            img=cv2.resize(img,pad.no_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)
        elif hf=="t":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)

        elif hf=="b":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
        elif hf=="bt" or hf=="tb":
            img=cv2.resize(img,pad.double_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        cimgs.append(img)

    img=np.concatenate(cimgs,axis=1)
    return img


def createFontImageFromComps(font,comps):
    '''
        creates font-space target images
        args:
            font    :   the font to use
            comps   :   the list of graphemes
        return:
            non-pad-corrected raw binary target
    '''
    
    # draw text
    image = PIL.Image.new(mode='L', size=font.getsize("".join(comps)))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text="".join(comps), fill=255, font=font)
    # reverse
    img=np.array(image)
    idx=np.where(img>0)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    img=img[y_min:y_max,x_min:x_max]
    return img    
    
def createRandomDictionary(valid_graphemes,num_samples,include_space=True):
    '''
        creates a randomized dictionary
        args:
            valid_graphemes :       list of graphemes that can be used to create a randomized dictionary 
            num_samples     :       number of data to be created if no dictionary is provided
            dict_max_len    :       the maximum length of data for randomized dictionary
            dict_min_len    :       the minimum length of data for randomized dictionary
        returns:
            a dictionary dataframe with "word" and "graphemes"
    '''
    word=[]
    graphemes=[]
    for _ in tqdm(range(num_samples)):
        len_word=random.choices(population=[1,2,3,4,5,6,7,8,9,10],weights=[0.05,0.05,0.1,0.15,0.15,0.15,0.15,0.1,0.05,0.05],k=1)[0]
        _graphemes=[]
        _space_added=False
        for _ in range(len_word):
            if include_space:
                # space
                if random_exec(weights=[0.8,0.2],match=1) and not _space_added:
                    num_space=random.randint(0,3)
                    if num_space>0:
                        for _ in range(num_space):
                            _graphemes.append(" ")
                    _space_added=True
            # grapheme
            _graphemes.append(random.choice(valid_graphemes))
        graphemes.append(_graphemes)
        word.append("".join(_graphemes))
    df=pd.DataFrame({"word":word,"graphemes":graphemes})
    return df 



#--------------------
# ops
#--------------------
def createSyntheticData(iden,
                        save_dir,
                        data_type,    
                        data_dir,
                        language,
                        img_dim,
                        num_samples=100000,
                        comp_dim=64,
                        pad_height=20,
                        use_only_graphemes=False,
                        use_only_numbers=False,
                        use_all=True,
                        fname_offset=0,
                        create_scene_data=True,
                        exclude_punct=False,
                        include_space=False):
    '''
        creates: 
            * handwriten word image
            * fontspace word image
            * a dataframe/csv that holds word level groundtruth
    '''
    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    LOG_INFO(save_dir)
    # save_paths
    class save:    
        img  =create_dir(save_dir,"images")
        amask=create_dir(save_dir,"attention_masks")
        imask=create_dir(save_dir,"image_masks")
        std  =create_dir(save_dir,"standards")
        csv=os.path.join(save_dir,"data.csv")
        txt=os.path.join(save_dir,"data.txt")
    
    
    # dataset
    if data_type=="printed":
        ds=DataSet(data_dir,language.iden,use_printed_only=True)
        pad=None
        if use_all:
            valid_graphemes=language.graphemes
        elif use_only_graphemes:
            valid_graphemes=language.dict_graphemes
        elif use_only_numbers:
            valid_graphemes=language.numbers
        if exclude_punct:
            valid_graphemes=[grapheme for grapheme in valid_graphemes if grapheme not in language.punctuations]
        include_space=include_space
    else:
        include_space=False
        ds=DataSet(data_dir,language.iden)
        if use_all:
            valid_graphemes=ds.valid_graphemes
        elif use_only_graphemes:
            valid_graphemes=ds.graphemes_list
        elif use_only_numbers:
            valid_graphemes=ds.numbers_list
        # pad
        class pad:
            no_pad_dim      =(comp_dim,comp_dim)
            single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
            double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
            top             =language.top_exts
            bot             =language.bot_exts
            height          =pad_height   

    # save data
    dictionary=createRandomDictionary(valid_graphemes,num_samples,include_space=include_space)
    # dataframe vars
    filepaths=[]
    words=[]
    fiden=0+fname_offset
    def_font=PIL.ImageFont.truetype(ds.def_font,comp_dim)
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            if data_type=="printed":
                fsize=random.randint(12,256)
                font=PIL.ImageFont.truetype(random.choice(ds.fonts),fsize)
                
                # image
                img=createFontImageFromComps(font,comps) 
                img=post_process_word_image(img)
                img=np.squeeze(img)
                # std
                std=createFontImageFromComps(def_font,comps) 
                
                
                if create_scene_data:
                    # extend image
                    if random_exec(match=1):
                        hi,wi=img.shape
                        ptype=random.choice(["tb","lr",None])
                        pdim=math.ceil(0.01*wi*random.randint(1,10))
                        img=padAllAround(img,pdim,0,pad_single=ptype)
                        
                    hi,wi=img.shape
                    back=cv2.imread(random.choice(ds.backs))
                    back=cv2.resize(back,(int(20*wi),int(20*hi)))
                    hb,wb,_=back.shape
                    x=random.randint(0,wb-wi)
                    y=random.randint(0,hb-hi)
                    back=back[y:y+hi,x:x+wi]
                    back[img==255]=randColor()
                    # imask
                    imask=np.copy(img)
                    imask[imask>0]=255 
                    # img
                    img=np.copy(back)
                    
                else:
                    # imask
                    imask=np.copy(img)
                    imask[imask>0]=255 
                    # img  
                    img=255-img
                    img=paper_noise(img) 
                    
                
            else:
                # std
                std=createFontImageFromComps(def_font,comps) 
                
                img_height=random.randint(12,128)
                # image
                img=createImgFromComps(df=ds.df,comps=comps,pad=pad)
                img=255-img
                h,w=img.shape
                w_new=int(img_height* w/h) 
                img=cv2.resize(img,(w_new,img_height))
                img=post_process_word_image(img)
                img=np.squeeze(img)
                # imask
                imask=np.copy(img)
                imask[imask>0]=255 
                
                # img
                img=255-img
                img=paper_noise(img)

            
            # save
            fname=f"{fiden}.png"
            # img
            img,mval=correctPadding(img,img_dim,ptype="left")
            img=noise.noise(img)
            cv2.imwrite(os.path.join(save.img,fname),img)
            
            # std
            std=cv2.merge((std,std,std))
            std,_=correctPadding(std,img_dim,ptype="left")
            cv2.imwrite(os.path.join(save.std,fname),std)
            
            # imask
            imask=cv2.merge((imask,imask,imask))
            imask,_=correctPadding(imask,img_dim,ptype="left")
            cv2.imwrite(os.path.join(save.imask,fname),imask)

            # amask
            h,w,d=img.shape
            amask=np.zeros((h,w))
            amask[:,mval:]=255
            cv2.imwrite(os.path.join(save.amask,fname),amask)
            

            filepaths.append(os.path.join(save.img,fname))
            word="".join(comps)
            words.append(word)
            fiden+=1
            with open(save.txt,"a+") as f:
                f.write(f"{fiden}.png#,#{word}#\n")
        except Exception as e:
           LOG_INFO(e)
    
    df=pd.DataFrame({"filepath":filepaths,"word":words})
    df.to_csv(os.path.join(save.csv),index=False)
    return df,save.csv