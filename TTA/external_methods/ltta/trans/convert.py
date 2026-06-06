import numpy as np
from PIL import Image
import pywt
import os
import random
import matplotlib.pyplot as plt
import csv
import functions as fun
import shutil

csv_file_name = 'data.csv'
original_img_folder = '/home/oem/servers/datasets/classification/cifar10/val/' 
noise_img_folder = '/home/oem/servers/datasets/classification/cifar10_cor/val/'  

def get_ttype(seq):
    if ttype == 1:
        name = 'Gaussian Noise'
    elif ttype == 2:
        name = 'Shot Noise'
    elif ttype == 3:
        name = 'Impulse Noise'
    elif ttype == 4:
        name = 'Defocus Blur'
    elif ttype == 5:
        name = 'Glass Blur'
    elif ttype == 6:
        name = 'Motion Blur'
    elif ttype == 7:
        name = 'Zoom Blur'
    elif ttype == 8:
        name = 'Snow'
    elif ttype == 9:
        name = 'Frost'
    elif ttype == 10:
        name = 'Fog'
    elif ttype == 11:
        name = 'Brightness'
    elif ttype == 12:
        name = 'Contrast'
    elif ttype == 13:
        name = 'Elastic'
    elif ttype == 14:
        name = 'Pixelate'
    else:
        name = 'JPEG'
    return name

def get_name(seq):
    if seq == 1:
        name = 'Gaussian_Noise' #
    elif seq == 2:
        name = 'Shot_Noise' #
    elif seq == 3:
        name = 'Impulse_Noise' #
    elif seq == 4:
        name = 'Defocus_Blur' #
    elif seq == 5:
        name = 'Glass_Blur' #
    elif seq == 6:
        name = 'Motion_Blur' #
    elif seq == 7:
        name = 'Zoom_Blur' #
    elif seq == 8:
        name = 'Snow' #
    elif seq == 9:
        name = 'Frost' #
    elif seq == 10:
        name = 'Fog' #
    elif seq == 11:
        name = 'Brightness'#
    elif seq == 12:
        name = 'Contrast' #
    elif seq == 13:
        name = 'Elastic' #
    elif seq == 14:
        name = 'Pixelate' #
    else:
        name = 'JPEG' #
    return str(name)

def get_noise_dir(seq):
    noise_name = get_name(seq)
    noise_img_folder = '/home/oem/servers/datasets/classification/cifar10_'+noise_name+'/val/' 
    return noise_img_folder



'''
1. CSV 구조 설계
2. Data 1. Noise Transform
3. Data 2. DWT Transform
'''

''' CSV frame 설계
1. 4^level 개의 데이터 프레임이 변환 전/후로 존재한다. = 4^level * 2 (개)
2. 전: 100 - Loss
2. 후: 100 - Loss
'''

'''
for x in range(1,10):
    print(int(random.uniform(1,10)))
'''
#for noise_seq in range(15):
for noise_seq in range(1):
    noise_seq = noise_seq + 12
    noise_img_folder = get_noise_dir(noise_seq)
    classes = list()

    original_img_folder_lst = os.listdir(original_img_folder)

    for item in original_img_folder_lst:
        classes.append(item)
    for clss in classes:
        #type(clss)
        #type(noise_img_folder)
        #print(clss)
        
        i_path = str(noise_img_folder) + str(clss)
        if os.path.exists(i_path):
            shutil.rmtree(i_path)
            os.makedirs(i_path, exist_ok=True)

        else:
            os.makedirs(i_path, exist_ok=True)



# length    
#print(len(os.listdir(original_img_folder + classes[0])))

    for clss in classes:
        images = os.listdir(original_img_folder + clss)
        print(len(images))
        for img_path in images:

            if img_path.split('.')[-1] != 'png':
                continue

            #convert 
            #ttype = int(random.uniform(1,16))
            ttype = int(noise_seq)
            #level = int(random.uniform(1,5))
            level = int(3)
            function_name = get_ttype(ttype)
            #print(function_name)
            func = fun.get_function(function_name)
            img = Image.open(original_img_folder + clss + '/' + img_path)
            trn_img = func(img, level)
            #print(type(trn_img))
            #print(trn_img.shape)
            save_img = Image.fromarray(np.uint8(trn_img), mode='RGB')
            #save_img.save(noise_img_folder + clss + '/' + img_path + '.png')
            save_img.save(str(noise_img_folder) + str(clss) + '/' + str(img_path))
        
        
    print(f'Save - {noise_seq}')

