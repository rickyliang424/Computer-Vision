# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:31:52 2023
@author: liang
"""
import os
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def preprocessing(filepath):
    img = np.array(Image.open(filepath))
    h, w = img.shape[0], img.shape[1]
    img = img[int(0.01*h):int(0.99*h),int(0.01*w):int(0.99*w)]
    img_1 = img[0*int(img.shape[0]/3):1*int(img.shape[0]/3),:]
    img_2 = img[1*int(img.shape[0]/3):2*int(img.shape[0]/3),:]
    img_3 = img[2*int(img.shape[0]/3):3*int(img.shape[0]/3),:]
    return img_1, img_2, img_3

def find_displacement(img_base, img_shift_1, img_shift_2, window, offset=np.zeros((2,2)).astype(int)):
    NCC_best_1 = -np.inf
    NCC_best_2 = -np.inf
    img_base_norm = (img_base-img_base.mean()) / np.linalg.norm(img_base)
    for column in range(offset[0,0]-window,offset[0,0]+window+1):
        for row in range(offset[0,1]-window,offset[0,1]+window+1):
            img_roll_1 = np.roll(img_shift_1, (column,row), axis=[1,0])
            img_roll_1_norm = (img_roll_1-img_roll_1.mean()) / np.linalg.norm(img_roll_1)
            NCC_1 = np.sum(img_roll_1_norm * img_base_norm)
            if NCC_1 > NCC_best_1:
                NCC_best_1 = NCC_1
                displacement_1 = [column,row]
    for column in range(offset[1,0]-window,offset[1,0]+window+1):
        for row in range(offset[1,1]-window,offset[1,1]+window+1):
            img_roll_2 = np.roll(img_shift_2, (column,row), axis=[1,0])
            img_roll_2_norm = (img_roll_2-img_roll_2.mean()) / np.linalg.norm(img_roll_2)
            NCC_2 = np.sum(img_roll_2_norm * img_base_norm)
            if NCC_2 > NCC_best_2:
                NCC_best_2 = NCC_2
                displacement_2 = [column,row]
    return displacement_1, displacement_2

def plot_all(img_base, img_align_1, img_align_2):
    fig = plt.figure(figsize=(9,6))
    fig.add_subplot(2,3,1)
    plt.imshow(np.stack((img_base, img_align_1, img_align_2), axis=2))
    plt.title(" c_order=[R,G,B]")
    plt.axis('off')
    fig.add_subplot(2,3,2)
    plt.imshow(np.stack((img_base, img_align_2, img_align_1), axis=2))
    plt.title(" c_order=[R,B,G]")
    plt.axis('off')
    fig.add_subplot(2,3,3)
    plt.imshow(np.stack((img_align_1, img_base, img_align_2), axis=2))
    plt.title(" c_order=[G,R,B]")
    plt.axis('off')
    fig.add_subplot(2,3,4)
    plt.imshow(np.stack((img_align_2, img_base, img_align_1), axis=2))
    plt.title(" c_order=[G,B,R]")
    plt.axis('off')
    fig.add_subplot(2,3,5)
    plt.imshow(np.stack((img_align_1, img_align_2, img_base), axis=2))
    plt.title(" c_order=[B,R,G]")
    plt.axis('off')
    fig.add_subplot(2,3,6)
    plt.imshow(np.stack((img_align_2, img_align_1, img_base), axis=2))
    plt.title(" c_order=[B,G,R]")
    plt.axis('off')
    return

def colorize(img_base, img_shift_1, img_shift_2, displacement_1, displacement_2, c_order=None):
    img_align_1 = np.roll(img_shift_1, displacement_1, axis=[1,0])
    img_align_2 = np.roll(img_shift_2, displacement_2, axis=[1,0])
    if c_order == None:
        plot_all(img_base, img_align_1, img_align_2) # choose the most plausible image and copy its color order.
    elif c_order == ['R','G','B']:
        img_final = np.stack((img_base, img_align_1, img_align_2), axis=2)
    elif c_order == ['R','B','G']:
        img_final = np.stack((img_base, img_align_2, img_align_1), axis=2)
    elif c_order == ['G','R','B']:
        img_final = np.stack((img_align_1, img_base, img_align_2), axis=2)
    elif c_order == ['G','B','R']:
        img_final = np.stack((img_align_2, img_base, img_align_1), axis=2)
    elif c_order == ['B','R','G']:
        img_final = np.stack((img_align_1, img_align_2, img_base), axis=2)
    elif c_order == ['B','G','R']:
        img_final = np.stack((img_align_2, img_align_1, img_base), axis=2)
    return img_final if c_order != None else None

def resize(img_1, img_2, img_3, m):
    img_1_new = np.array(Image.fromarray(img_1).resize((int(img_1.shape[0]*m), int(img_1.shape[1]*m))))
    img_2_new = np.array(Image.fromarray(img_2).resize((int(img_2.shape[0]*m), int(img_2.shape[1]*m))))
    img_3_new = np.array(Image.fromarray(img_3).resize((int(img_3.shape[0]*m), int(img_3.shape[1]*m))))
    return img_1_new, img_2_new, img_3_new

def image_pyramid(img_1, img_2, img_3):
    img_1_x02, img_2_x02, img_3_x02 = resize(img_1, img_2, img_3, 1/2)
    img_1_x04, img_2_x04, img_3_x04 = resize(img_1, img_2, img_3, 1/4)
    img_1_x08, img_2_x08, img_3_x08 = resize(img_1, img_2, img_3, 1/8)
    img_1_x16, img_2_x16, img_3_x16 = resize(img_1, img_2, img_3, 1/16)
    displ_1_x16, displ_2_x16 = find_displacement(img_1_x16, img_2_x16, img_3_x16, 15)
    displ_1_x08, displ_2_x08 = find_displacement(img_1_x08, img_2_x08, img_3_x08, 10, 2*np.array([displ_1_x16, displ_2_x16]))
    displ_1_x04, displ_2_x04 = find_displacement(img_1_x04, img_2_x04, img_3_x04, 10, 2*np.array([displ_1_x08, displ_2_x08]))
    displ_1_x02, displ_2_x02 = find_displacement(img_1_x02, img_2_x02, img_3_x02, 10, 2*np.array([displ_1_x04, displ_2_x04]))
    displ_1, displ_2 = find_displacement(img_1, img_2, img_3, 10, 2*np.array([displ_1_x02, displ_2_x02]))
    return displ_1, displ_2

#%% Check the correctness of the code
img_c = np.array(Image.open("C:/Users/liang/Desktop/check.png"))
img_R = np.full((550,550),255,dtype=np.uint8)
img_G = np.full((550,550),255,dtype=np.uint8)
img_B = np.full((550,550),255,dtype=np.uint8)
img_R[4:4+len(img_c),8:8+len(img_c)] = img_c[:,:,0]
img_G[10:10+len(img_c),5:5+len(img_c)] = img_c[:,:,1]
img_B[15:15+len(img_c),15:15+len(img_c)] = img_c[:,:,2]
displacement_1, displacement_2 = find_displacement(img_R, img_G, img_B, 15)
img_final = colorize(img_R, img_G, img_B, displacement_1, displacement_2, ['R','G','B'])
Image.fromarray(img_c).show()
Image.fromarray(img_final).show()

#%% Basic alignment - Determine color order of each image
path = "C:/Users/liang/Desktop/data/"
for file in os.listdir(path):
    img_1, img_2, img_3 = preprocessing(path + file)
    displacement_1, displacement_2 = find_displacement(img_1, img_2, img_3, 15)
    img_final = colorize(img_1, img_2, img_3, displacement_1, displacement_2)

#%% Basic alignment - Registration and find displacement
savepath = "C:/Users/liang/Desktop/results/basic_alignment/"
for file in os.listdir(path):
    img_1, img_2, img_3 = preprocessing(path + file)
    print('\n' + '=' * 50)
    print('filename is', file)
    displacement_1, displacement_2 = find_displacement(img_1, img_2, img_3, 15)
    img_final = colorize(img_1, img_2, img_3, displacement_1, displacement_2, ['B','G','R'])
    Image.fromarray(img_final).save(savepath + file[:-4] + '_B-based.jpg')
    print('\nBase channel: Blue')
    print('Green channel offset:', displacement_1)
    print('Red channel offset:', displacement_2)
    displacement_1, displacement_2 = find_displacement(img_2, img_1, img_3, 15)
    img_final = colorize(img_2, img_1, img_3, displacement_1, displacement_2, ['G','B','R'])
    Image.fromarray(img_final).save(savepath + file[:-4] + '_G-based.jpg')
    print('\nBase channel: Green')
    print('Blue channel offset:', displacement_1)
    print('Red channel offset:', displacement_2)
    displacement_1, displacement_2 = find_displacement(img_3, img_1, img_2, 15)
    img_final = colorize(img_3, img_1, img_2, displacement_1, displacement_2, ['R','B','G'])
    Image.fromarray(img_final).save(savepath + file[:-4] + '_R-based.jpg')
    print('\nBase channel: Red')
    print('Blue channel offset:', displacement_1)
    print('Green channel offset:', displacement_2)

#%% Multiscale alignment
path = "C:/Users/liang/Desktop/data_hires/"
savepath = "C:/Users/liang/Desktop/results/multiscale_alignment/"
for file in os.listdir(path):
    img_1, img_2, img_3 = preprocessing(path + file)
    print('\n' + '=' * 50)
    print('filename is', file)
    start_time = time.time()
    displ_1, displ_2 = image_pyramid(img_1, img_2, img_3)
    img_uint16 = colorize(img_1, img_2, img_3, displ_1, displ_2, ['B','G','R'])
    img_final = (img_uint16 / 65535 * 255).astype(np.uint8)
    end_time = time.time()
    Image.fromarray(img_final).save(savepath + file[:-4] + '_B-based.jpg')
    print('\nBase channel: Blue')
    print('Green channel offset:', displ_1)
    print('Red channel offset:', displ_2)
    print('Total compute time:', end_time - start_time)
    start_time = time.time()
    displ_1, displ_2 = image_pyramid(img_2, img_1, img_3)
    img_uint16 = colorize(img_2, img_1, img_3, displ_1, displ_2, ['G','B','R'])
    img_final = (img_uint16 / 65535 * 255).astype(np.uint8)
    end_time = time.time()
    Image.fromarray(img_final).save(savepath + file[:-4] + '_G-based.jpg')
    print('\nBase channel: Green')
    print('Blue channel offset:', displ_1)
    print('Red channel offset:', displ_2)
    print('Total compute time:', end_time - start_time)
    start_time = time.time()
    displ_1, displ_2 = image_pyramid(img_3, img_1, img_2)
    img_uint16 = colorize(img_3, img_1, img_2, displ_1, displ_2, ['R','B','G'])
    img_final = (img_uint16 / 65535 * 255).astype(np.uint8)
    end_time = time.time()
    Image.fromarray(img_final).save(savepath + file[:-4] + '_R-based.jpg')
    print('\nBase channel: Red')
    print('Blue channel offset:', displ_1)
    print('Green channel offset:', displ_2)
    print('Total compute time:', end_time - start_time)
