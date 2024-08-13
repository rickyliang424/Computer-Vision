# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:44:24 2023
@author: liang
"""
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def SSD(mtx1, mtx2):
    ssd = np.sum(np.square(mtx1 - mtx2))
    return ssd

def SAD(mtx1, mtx2):
    sad = np.sum(np.abs(mtx1 - mtx2))
    return sad

def NCC(mtx1, mtx2):
    mtx1_norm = (mtx1 - np.mean(mtx1)).astype(np.float32)
    mtx2_norm = (mtx2 - np.mean(mtx2)).astype(np.float32)
    ncc = np.sum(mtx1_norm * mtx2_norm) / ((np.sqrt(np.sum(mtx1_norm**2)) * np.sqrt(np.sum(mtx2_norm**2))) + 1e-9)
    return ncc

def calc_disparity(img_L, img_R, window_size, search_range, func):
    start_time = time.time()
    function = {'SSD': SSD, 'SAD': SAD, 'NCC': NCC}
    disp_mtx = []
    for row in tqdm(range(img_L.shape[0] - window_size)):
        disparity = []
        for col1 in range(img_L.shape[1] - window_size):
            mtx1 = img_L[row:(row + window_size), col1:(col1 + window_size)].flatten()
            init = 0 if col1 < search_range else col1 - search_range
            scores = []
            for col2 in range(col1, init-1, -1):
                mtx2 = img_R[row:(row + window_size), col2:(col2 + window_size)].flatten()
                scores.append(function[func](mtx1, mtx2))
            if func == 'NCC':
                disparity.append(np.argmax(scores))
            else:
                disparity.append(np.argmin(scores))
        disp_mtx.append(disparity)
    disp_mtx = np.array(disp_mtx)
    end_time = time.time()
    runtime = end_time - start_time
    print('Elapsed Time: ' + str(runtime) + ' s')
    return disp_mtx, round(runtime,2)

# filename = 'tsukuba'
filename = 'moebius'

window_size = 75
search_range = 55
function = 'SSD'
# function = 'SAD'
# function = 'NCC'

img_L_path = './part2_data/' + filename + '1.png'
img_R_path = './part2_data/' + filename + '2.png'
img_L = np.array(Image.open(img_L_path).convert('L'), dtype='int64')
img_R = np.array(Image.open(img_R_path).convert('L'), dtype='int64')
disp_mtx, runtime = calc_disparity(img_L, img_R, window_size, search_range, function)

fig, ax = plt.subplots(figsize=(8,6))
heatmap = ax.imshow(disp_mtx, cmap='viridis', aspect='auto')
cbar = plt.colorbar(heatmap, ax=ax)
plt.title('w={}, r={}, f={}, t={}'.format(window_size, search_range, function, runtime))
plt.show()
# plt.savefig('./results/' + '{}_w{}_s{}_{}.png'.format(filename, window_size, search_range, function))
