# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 17:31:52 2023
@author: liang
"""
import os
import time
import scipy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#%% Try croping
# # path = "C:/Users/liang/Desktop/data/"
# # path = "C:/Users/liang/Desktop/data_hires/"
# for file in os.listdir(path):
#     img = np.array(Image.open(path + file))
#     v_mean = np.mean(img, axis=0)
#     h_mean = np.mean(img, axis=1)
#     vx = np.linspace(0, 100, len(v_mean))
#     hx = np.linspace(0, 100, len(h_mean))
#     plt.figure()
#     # plt.plot(vx, v_mean / np.max(v_mean) * 100)
#     # plt.plot(hx, h_mean / np.max(h_mean) * 100)
#     plt.xticks(np.linspace(0, 100, 10+1))
#     plt.yticks(np.linspace(0, 100, 10+1))
#     plt.grid()

#%% Function defining
def preprocessing(filepath, show=False, save=None):
    img = np.array(Image.open(filepath))
    img = (img / 65535 * 255).astype(np.uint8) if img.dtype.name=='uint16' else img
    h_mean_ud, h_mean_du = np.mean(img, axis=1), np.mean(img, axis=1)[::-1]
    v_mean_lr, v_mean_rl = np.mean(img, axis=0), np.mean(img, axis=0)[::-1]
    new_border = []
    for vector in [h_mean_ud, h_mean_du, v_mean_lr, v_mean_rl]:
        count = 0
        for i in range(len(vector)):
            if vector[i] <= 0.25*np.max(img):
                count = 1
            if count == 1 and vector[i] >= 0.25*np.max(img):
                new_border.append(i)
                break
    new_img = img[new_border[0]:-new_border[1], new_border[2]:-new_border[3]]
    img_1 = new_img[0*int(new_img.shape[0]/3):1*int(new_img.shape[0]/3), :]
    img_2 = new_img[1*int(new_img.shape[0]/3):2*int(new_img.shape[0]/3), :]
    img_3 = new_img[2*int(new_img.shape[0]/3):3*int(new_img.shape[0]/3), :]
    
    img_show = np.stack([img, img, img], axis=2)
    img_show[new_border[0]-1:new_border[0]+1, :, :] = 0
    img_show[new_border[0]-1:new_border[0]+1, :, 0] = 255
    img_show[1*int(new_img.shape[0]/3)-1:1*int(new_img.shape[0]/3)+1, :, :] = 0
    img_show[1*int(new_img.shape[0]/3)-1:1*int(new_img.shape[0]/3)+1, :, 0] = 255
    img_show[2*int(new_img.shape[0]/3)-1:2*int(new_img.shape[0]/3)+1, :, :] = 0
    img_show[2*int(new_img.shape[0]/3)-1:2*int(new_img.shape[0]/3)+1, :, 0] = 255
    img_show[-new_border[1]-1:-new_border[1]+1, :, :] = 0
    img_show[-new_border[1]-1:-new_border[1]+1, :, 0] = 255
    img_show[:, new_border[2]-1:new_border[2]+1, :] = 0
    img_show[:, new_border[2]-1:new_border[2]+1, 0] = 255
    img_show[:, -new_border[3]-1:-new_border[3]+1, :] = 0
    img_show[:, -new_border[3]-1:-new_border[3]+1, 0] = 255
    if show == True:
        Image.fromarray(img_show).show()    
    if save != None:
        Image.fromarray(img_show).save(savepath + 'Preprocessed_' + save + '.jpg')
    return img_1, img_2, img_3

def LoG_filtering(images, show=False, save=None):
    filtered_images = []
    for img in images:
        img_f = scipy.ndimage.gaussian_laplace(img, sigma=1)
        filtered_images.append(img_f)
    if show == True:
        for i in range(len(images)):
            Image.fromarray(np.concatenate((images[i], filtered_images[i]), axis=1)).show()
    if save != None:
        for i in range(len(filtered_images)):
            Image.fromarray(filtered_images[i]).save(savepath + 'Filtered_' + save + '_' + str(i) + '.jpg')
    return filtered_images

def find_displacement(img_base, img_shift_1, img_shift_2, c_order, show=False, save=None):
    img_base_norm = (img_base - img_base.mean()) / np.linalg.norm(img_base)
    img_shift_1_norm = (img_shift_1 - img_shift_1.mean()) / np.linalg.norm(img_shift_1)
    img_shift_2_norm = (img_shift_2 - img_shift_2.mean()) / np.linalg.norm(img_shift_2)
    FT_b = np.fft.fftshift(np.fft.fft2(img_base_norm))
    FT_1 = np.fft.fftshift(np.fft.fft2(img_shift_1_norm))
    FT_2 = np.fft.fftshift(np.fft.fft2(img_shift_2_norm))
    FT_b_1c = FT_b * np.conjugate(FT_1)
    FT_b_2c = FT_b * np.conjugate(FT_2)
    NCC_1 = np.fft.ifft2(FT_b_1c)
    NCC_2 = np.fft.ifft2(FT_b_2c)
    
    NCC_best_1 = np.abs(NCC_1).argmax()
    NCC_best_2 = np.abs(NCC_2).argmax()
    displacement_1 = list(np.unravel_index(NCC_best_1, NCC_1.shape))
    displacement_2 = list(np.unravel_index(NCC_best_2, NCC_2.shape))
    if displacement_1[0] > 0.5*img_base.shape[0]:
        displacement_1[0] = displacement_1[0] - img_base.shape[0]
    if displacement_1[1] > 0.5*img_base.shape[1]:
        displacement_1[1] = displacement_1[1] - img_base.shape[1]
    if displacement_2[0] > 0.5*img_base.shape[0]:
        displacement_2[0] = displacement_2[0] - img_base.shape[0]
    if displacement_2[1] > 0.5*img_base.shape[1]:
        displacement_2[1] = displacement_2[1] - img_base.shape[1]
    
    if show == True:
        i = 1
        for ift in [NCC_1, NCC_2]:
            plt.figure(figsize=(8,6))
            plt.imshow(np.log(abs(ift)))
            title = 'Inverse Fourier Transform of ' + c_order[i] + ' to ' + c_order[0] + ' Alignment'
            plt.title(' ' + title + '\n', fontsize=16)
            plt.colorbar()
            plt.tight_layout()
            if save != None:
                plt.savefig(savepath + 'iFT_' + save + '_' + c_order[i] + '_to_' + c_order[0] + '_alignment.jpg')
            i = i + 1
    return displacement_1, displacement_2

def colorize(img_base, img_shift_1, img_shift_2, displacement_1, displacement_2, c_order):
    img_align_1 = np.roll(img_shift_1, displacement_1, axis=[0,1])
    img_align_2 = np.roll(img_shift_2, displacement_2, axis=[0,1])
    if   c_order == 'RGB':
        img_final = np.stack((img_base, img_align_1, img_align_2), axis=2)
    elif c_order == 'RBG':
        img_final = np.stack((img_base, img_align_2, img_align_1), axis=2)
    elif c_order == 'GRB':
        img_final = np.stack((img_align_1, img_base, img_align_2), axis=2)
    elif c_order == 'GBR':
        img_final = np.stack((img_align_2, img_base, img_align_1), axis=2)
    elif c_order == 'BRG':
        img_final = np.stack((img_align_1, img_align_2, img_base), axis=2)
    elif c_order == 'BGR':
        img_final = np.stack((img_align_2, img_align_1, img_base), axis=2)
    return img_final

#%% Check the correctness of the code
# img_c = np.array(Image.open("C:/Users/liang/Desktop/check.png"))
# img_R = np.full((550,550),255,dtype=np.uint8)
# img_G = np.full((550,550),255,dtype=np.uint8)
# img_B = np.full((550,550),255,dtype=np.uint8)
# img_R[4:4+len(img_c),8:8+len(img_c)] = img_c[:,:,0]
# img_G[10:10+len(img_c),5:5+len(img_c)] = img_c[:,:,1]
# img_B[15:15+len(img_c),15:15+len(img_c)] = img_c[:,:,2]
# displacement_1, displacement_2 = find_displacement(img_R, img_G, img_B, 'RBG', show=False, save=None)
# img_final = colorize(img_R, img_G, img_B, displacement_1, displacement_2, c_order='RGB')
# Image.fromarray(img_c).show()
# Image.fromarray(img_final).show()

#%% Fourier-based color channel alignment
# path = "C:/Users/liang/Desktop/data/"
path = "C:/Users/liang/Desktop/data_hires/"
savepath = "C:/Users/liang/Desktop/MP2_results/"
for file in os.listdir(path):
    filepath = path + file
    filename = file[:-4]
    img_1, img_2, img_3 = preprocessing(filepath, show=False, save=filename)
    img_1f, img_2f, img_3f = preprocessing(filepath, show=False, save=None)
    # img_1f, img_2f, img_3f = LoG_filtering([img_1, img_2, img_3], show=False, save=filename)
    print('\n' + '=' * 50)
    print('filename is', file)
    
    start_time = time.time()
    displacement_1, displacement_2 = find_displacement(img_1f, img_2f, img_3f, 'BGR', show=True, save=filename)
    img_final = colorize(img_1, img_2, img_3, displacement_1, displacement_2, c_order='BGR')
    end_time = time.time()
    Image.fromarray(img_final).save(savepath + file[:-4] + '_B-based.jpg')
    print('\nBase channel: Blue')
    print('Green channel offset:', displacement_1)
    print('Red channel offset:', displacement_2)
    print('Total compute time:', end_time - start_time)
    
    start_time = time.time()
    displacement_1, displacement_2 = find_displacement(img_2f, img_1f, img_3f, 'GBR', show=True, save=filename)
    img_final = colorize(img_2, img_1, img_3, displacement_1, displacement_2, c_order='GBR')
    end_time = time.time()
    Image.fromarray(img_final).save(savepath + file[:-4] + '_G-based.jpg')
    print('\nBase channel: Green')
    print('Blue channel offset:', displacement_1)
    print('Red channel offset:', displacement_2)
    print('Total compute time:', end_time - start_time)
    
    start_time = time.time()
    displacement_1, displacement_2 = find_displacement(img_3f, img_1f, img_2f, 'RBG', show=True, save=filename)
    img_final = colorize(img_3, img_1, img_2, displacement_1, displacement_2, c_order='RBG')
    end_time = time.time()
    Image.fromarray(img_final).save(savepath + file[:-4] + '_R-based.jpg')
    print('\nBase channel: Red')
    print('Blue channel offset:', displacement_1)
    print('Green channel offset:', displacement_2)
    print('Total compute time:', end_time - start_time)
