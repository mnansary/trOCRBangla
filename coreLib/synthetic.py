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
    for comp in comps:
        cdf=df.loc[df.label==comp]
        cdf=cdf.sample(frac=1)
        if len(cdf)==1:
            img_paths.append(cdf.iloc[0,2])
        else:
            img_paths.append(cdf.iloc[random.randint(0,len(cdf)-1),2])
    
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
    
def createRandomDictionary(valid_graphemes,num_samples):
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
        for _ in range(len_word):
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
                        num_samples=100000,
                        comp_dim=64,
                        pad_height=20,
                        use_only_graphemes=False,
                        use_only_numbers=False,
                        use_all=True,
                        fname_offset=0,
                        return_df=False,
                        create_scene_data=True,
                        exclude_punct=False):
    '''
        creates: 
            * handwriten word image
            * fontspace word image
            * a dataframe/csv that holds word level groundtruth
        args:
            iden            :       identifier of the dataset
            img_height      :       height of the image
            save_dir        :       the directory to save the outputs
            data_type       :       the data_type to create (handwritten/printed)
            data_dir        :       the directory that holds graphemes and numbers and fonts data
            language        :       the specific language to use
            num_samples     :       number of data to be created 
            
    '''
    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    LOG_INFO(save_dir)
    # save_paths
    class save:    
        img=create_dir(save_dir,"images")
        csv=os.path.join(save_dir,"data.csv")
    
    # dataset
    if data_type=="printed":
        ds=DataSet(data_dir,language.iden,use_printed_only=True)
        pad=None
        if use_all:
            valid_graphemes=language.valid
        elif use_only_graphemes:
            valid_graphemes=language.dict_graphemes
        elif use_only_numbers:
            valid_graphemes=language.numbers
        if exclude_punct:
            valid_graphemes=[grapheme for grapheme in valid_graphemes if grapheme not in language.punctuations]
    else:
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
    dictionary=createRandomDictionary(valid_graphemes,num_samples)
    # dataframe vars
    filepaths=[]
    words=[]
    fiden=0+fname_offset
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            if data_type=="printed":
                fsize=random.randint(8,256)
                font=PIL.ImageFont.truetype(random.choice(ds.fonts),fsize)
                img=createFontImageFromComps(font,comps) 
                img=post_process_word_image(img)
                img=np.squeeze(img)
                if create_scene_data:
                    back=cv2.imread(random.choice(ds.backs))
                    back=cv2.resize(back,(int(20*fsize),int(20*fsize)))
                    hb,wb,_=back.shape
                    hi,wi=img.shape
                    x=random.randint(0,wb-wi)
                    y=random.randint(0,hb-hi)
                    back=back[y:y+hi,x:x+wi]
                    back[img==255]=randColor()
                    img=np.copy(back)
                else:
                    img=255-img   
                    img=cv2.merge((img,img,img))
                    img=noise.noise(img)
            else:
                img_height=random.randint(8,128)
                # image
                img=createImgFromComps(df=ds.df,comps=comps,pad=pad)
                img=255-img
                h,w=img.shape
                w_new=int(img_height* w/h) 
                img=cv2.resize(img,(w_new,img_height))
                img=post_process_word_image(img)
                img=np.squeeze(img)
                img=255-img
                img=cv2.merge((img,img,img))
                img=noise.noise(img)
                
            # save
            fname=f"{fiden}.png"
            cv2.imwrite(os.path.join(save.img,fname),img)
            filepaths.append(os.path.join(save.img,fname))
            words.append("".join(comps))
            fiden+=1
        except Exception as e:
           LOG_INFO(e)
    df=pd.DataFrame({"filepath":filepaths,"word":words})
    if return_df:
        return df,save.csv
    else:
        df.to_csv(os.path.join(save.csv),index=False)
        return save.csv