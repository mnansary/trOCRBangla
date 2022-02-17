#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import math
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path

def randColor(col=True):
    '''
        generates random color
    '''
    if col:
        return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    else:
        d=random.randint(0,64)
        return (d,d,d)

def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match
#--------------------
# processing 
#--------------------
def get_warped_image(img,warp_vec,coord,max_warp_perc=20):
    '''
        returns warped image and new coords
        args:
            img      : image to warp
            warp_vec : which vector to warp
            coord    : list of current coords
              
    '''
    height,width=img.shape
 
    # construct dict warp
    x1,y1=coord[0]
    x2,y2=coord[1]
    x3,y3=coord[2]
    x4,y4=coord[3]
    # warping calculation
    xwarp=random.randint(0,max_warp_perc)/100
    ywarp=random.randint(0,max_warp_perc)/100
    # construct destination
    dx=int(width*xwarp)
    dy=int(height*ywarp)
    # const
    if warp_vec=="p1":
        dst= [[dx,dy], [x2,y2],[x3,y3],[x4,y4]]
    elif warp_vec=="p2":
        dst=[[x1,y1],[x2-dx,dy],[x3,y3],[x4,y4]]
    elif warp_vec=="p3":
        dst= [[x1,y1],[x2,y2],[x3-dx,y3-dy],[x4,y4]]
    else:
        dst= [[x1,y1],[x2,y2],[x3,y3],[dx,y4-dy]]
    M   = cv2.getPerspectiveTransform(np.float32(coord),np.float32(dst))
    img = cv2.warpPerspective(img, M, (width,height),flags=cv2.INTER_NEAREST)
    return img,dst

def warp_data(img):
    warp_types=["p1","p2","p3","p4"]
    height,width=img.shape

    coord=[[0,0], 
        [width-1,0], 
        [width-1,height-1], 
        [0,height-1]]

    # warp
    for i in range(2):
        if i==0:
            idxs=[0,2]
        else:
            idxs=[1,3]
        if random_exec():    
            idx=random.choice(idxs)
            img,coord=get_warped_image(img,warp_types[idx],coord)
    return img


def rotate_image(mat, angle_max=5):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    angle=random.randint(-angle_max,angle_max)
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat

def post_process_word_image(img):
     # warp 20%
    if random_exec(weights=[0.2,0.8]):
        img=warp_data(img)
        return img
    # rotate/curve
    if random_exec(weights=[0.5,0.5]):
        img=rotate_image(img)
        return img
    return img

#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr

def removeShadow(img):
    '''
        removes shadows
    '''
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def padAllAround(img,pad_dim,pad_val,pad_single=None):
    '''
        pads all around the image
    '''
    if pad_single is None:
        h,w=img.shape
        # pads
        left_pad =np.ones((h,pad_dim))*pad_val
        right_pad=np.ones((h,pad_dim))*pad_val
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
        # shape
        h,w=img.shape
        top_pad =np.ones((pad_dim,w))*pad_val
        bot_pad=np.ones((pad_dim,w))*pad_val
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    elif pad_single=="tb":
        # shape
        h,w=img.shape
        top_pad =np.ones((pad_dim,w))*pad_val
        bot_pad=np.ones((pad_dim,w))*pad_val
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    else:
        h,w=img.shape
        # pads
        left_pad =np.ones((h,pad_dim))*pad_val
        right_pad=np.ones((h,pad_dim))*pad_val
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
    return img
#---------------------------------------------------------------
# parsing utils
#---------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

#----------------------------------------
# noise utils
#----------------------------------------
def gaussian_noise(height, width):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    return np.array(Image.fromarray(image).convert("RGB"))

def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = random.random() * 30 + 20  # frequency
    phase = random.random() * 2 * math.pi  # phase
    rotation_count = random.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return np.array(image.convert("RGB"))


class Modifier:
    def __init__(self,
                bi_filter_dim_min=7,
                bi_filter_dim_max=12,
                bi_filter_sigma_max=80,
                bi_filter_sigma_min=70,
                use_gaussblur=False,
                use_bifilter=False,
                use_medianblur=True,
                use_gaussnoise=False,
                use_brightness=False):

        self.bi_filter_dim_min      =   bi_filter_dim_min
        self.bi_filter_dim_max      =   bi_filter_dim_max
        self.bi_filter_sigma_min    =   bi_filter_sigma_min
        self.bi_filter_sigma_max    =   bi_filter_sigma_max
        
        self.use_brightness         =   use_brightness
        self.use_bifilter           =   use_bifilter
        self.use_gaussnoise         =   use_gaussnoise
        self.use_gaussblur          =   use_gaussblur
        self.use_medianblur         =   use_medianblur
        
    def __initParams(self):
        self.blur_kernel_size=3
        self.bi_filter_dim   =random.randrange(self.bi_filter_dim_min,
                                               self.bi_filter_dim_max, 
                                               2)
        self.bi_filter_sigma =random.randint(self.bi_filter_sigma_min,
                                             self.bi_filter_sigma_max)
        self.ops             =   [  self.__blur]
        if self.use_medianblur:
            self.ops.append(self.__medianBlur)
        if self.use_gaussblur:
            self.ops.append(self.__gaussBlur)
        if self.use_gaussnoise:
            self.ops.append(self.__gaussNoise)
        if self.use_bifilter:
            self.ops.append(self.__biFilter)
        if self.use_brightness:
            self.ops.append(self.__addBrightness)


    def __blur(self,img):
        return cv2.blur(img,
                        (self.blur_kernel_size,
                        self.blur_kernel_size),
                         0)
    def __gaussBlur(self,img):
        return cv2.GaussianBlur(img,
                                (self.blur_kernel_size,
                                self.blur_kernel_size),
                                0) 
    def __medianBlur(self,img):
        return  cv2.medianBlur(img,
                               self.blur_kernel_size)
    def __biFilter(self,img):
        return cv2.bilateralFilter(img,
                                   self.bi_filter_dim,
                                   self.bi_filter_sigma,
                                   self.bi_filter_sigma)

    def __gaussNoise(self,image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        image = image+gauss
        return image.astype("uint8")
    
    def __addBrightness(self,image):    
        ## Conversion to HLSmask
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)     
        image_HLS = np.array(image_HLS, dtype = np.float64)
        ## generates value between 0.5 and 1.5       
        random_brightness_coefficient = np.random.uniform()+0.5  
        ## scale pixel values up or down for channel 1(Lightness) 
        image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient
        ##Sets all values above 255 to 255    
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255     
        image_HLS = np.array(image_HLS, dtype = np.uint8)    
        ## Conversion to RGB
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)     
        return image_RGB
    
    def noise(self,img):
        self.__initParams()
        img=img.astype("uint8")
        idx = random.choice(range(len(self.ops)))
        img = self.ops.pop(idx)(img)
        return img
#---------------------wrapper
def paper_noise(img):
    if random_exec(weights=[0.8,0.2]):
        h,w=img.shape
        back_fn=random.choice([quasicrystal,gaussian_noise])
        back=back_fn(h,w)
        r=random.randint(0,25)
        g=random.randint(0,25)
        b=random.randint(0,25)
        back[img==0]=(r,g,b)
        img=np.copy(back)
    else:
        img=cv2.merge((img,img,img))
    return img