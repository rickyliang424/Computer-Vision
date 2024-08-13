# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 17:31:52 2023
@author: liang
"""
import cv2
import scipy
import skimage
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

def blob_construction(image, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=None):
    #%% preprocessing
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.float32(img_gray) / 255
    
    #%% find corner and show
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    mx = scipy.ndimage.maximum_filter(dst, size=size*2+1)
    dst_binary = (mx == dst) & (mx > threshold * np.max(dst))
    coord = np.array(np.where(dst_binary)).transpose()
    # fig, ax = plt.subplots(facecolor='#eeeeee', figsize=(20,15))
    # ax.set_aspect('equal')
    # ax.imshow(img_gray, cmap='gray')
    # for x, y in zip(coord[:,1], coord[:,0]):
    #     ax.add_patch(Circle((x, y), 2, color='r', linewidth=2, fill=False))
    # plt.title('%i corners' % len(coord[:,1]), fontsize=40)
    # plt.axis('off')
    
    #%% find scale
    response = np.zeros((len(coord), level))
    for i in range(level):
        scale = initial * factor ** i
        img_d = skimage.transform.resize(img, (int(img.shape[0]/scale),int(img.shape[1]/scale)), anti_aliasing=True)
        img_f = sigma * sigma * scipy.ndimage.gaussian_laplace(img_d, sigma)
        img_u = skimage.transform.resize(img_f, (img.shape[0], img.shape[1]), anti_aliasing=True)
        for j in range(len(coord)):
            response[j,i] = np.power((img_u[coord[j,0],coord[j,1]]), 2)
    max_scales = initial * factor ** np.argmax(response, axis=1)
    
    #%% find orientation
    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    gradient_x = cv2.filter2D(img, -1, kernel=sobel_x)
    gradient_y = cv2.filter2D(img, -1, kernel=sobel_y)
    angles = np.arctan2(gradient_y, gradient_x)
    orientation = np.zeros(len(coord))
    for i in range(len(coord)):
        scale = int(max_scales[i] / 2)
        u = coord[i,0]-scale if coord[i,0]-scale >= 0 else 0
        d = coord[i,0]+scale+1 if coord[i,0]+scale+1 <= angles.shape[0] else angles.shape[0]
        l = coord[i,1]-scale if coord[i,1]-scale >= 0 else 0
        r = coord[i,1]+scale+1 if coord[i,1]+scale+1 <= angles.shape[1] else angles.shape[1]
        window = angles[u:d, l:r]
        hist = np.unique(window // (np.pi/12), return_counts=True)
        orientation[i] = hist[0][np.argmax(hist[1])] * (np.pi/12)
    
    #%% plot result
    fig, ax = plt.subplots(facecolor='#eeeeee', figsize=(20,15))
    ax.set_aspect('equal')
    ax.imshow(img_gray, cmap='gray')
    for x, y, r, o in zip(coord[:,1], coord[:,0], max_scales/2, orientation):
        ax.plot(x, y, 'gx', markersize=8, markeredgewidth=2)
        ax.add_patch(Circle((x, y), r, color='r', linewidth=3, fill=False))
        ax.arrow(x, y, 0.5*r*np.cos(o), 0.5*r*np.sin(o), color='b', width=1, head_width=3)
    plt.title('%i circles' % len(coord[:,1]), fontsize=40)
    plt.xlim([0, image.shape[1]])
    plt.ylim([image.shape[0], 0])
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    if save != None:
        fig.savefig(save)
    
    #%%
    return

#%% show image
def show_image(img, title=None, save=None):
    fig, ax = plt.subplots(facecolor='#eeeeee', figsize=(20,15))
    ax.set_aspect('equal')
    ax.imshow(img[:,:,::-1])
    # ax.plot(int(img.shape[1]/2), int(img.shape[0]/2), 'ro', markeredgewidth=20) # check center
    if title != None:
        plt.title(title, fontsize=40)
    plt.axis('off')
    plt.tight_layout()
    if save != None:
        fig.savefig(save)
    return

#%% test
file = 'house.jpg'
# file = 'butterfly.jpg'
# file = 'fishes.jpg'
# file = 'einstein.jpg'
# file = 'sunflowers.jpg'

img_base = cv2.imread("C:/Users/liang/Desktop/images/" + file)
h, w = img_base.shape[0], img_base.shape[1]
img_sh_l = cv2.warpAffine(img_base, np.float32([[1,0,-int(0.2*w)], [0,1,0]]), (w,h))
img_sh_r = cv2.warpAffine(img_base, np.float32([[1,0,+int(0.2*w)], [0,1,0]]), (w,h))
img_ro_p = cv2.rotate(img_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_ro_n = cv2.rotate(img_base, cv2.ROTATE_90_CLOCKWISE)
img_en_2 = cv2.warpAffine(img_base, np.float32([[2,0,-int(w/2)], [0,2,-int(h/2)]]), (w,h))

savepath = 'C:/Users/liang/Desktop/MP2_results/'
savename = []
savename.append(file[:-4] + ' - Base image' + '.jpg')
savename.append(file[:-4] + ' - Image shifted 20% to the left' + '.jpg')
savename.append(file[:-4] + ' - Image shifted 20% to the right' + '.jpg')
savename.append(file[:-4] + ' - Image rotated by 90 degrees counterclockwise' + '.jpg')
savename.append(file[:-4] + ' - Image rotated by 90 degrees clockwise' + '.jpg')
savename.append(file[:-4] + ' - Image enlarged by a factor of 2' + '.jpg')

show_image(img_base, save=savepath+savename[0])
show_image(img_sh_l, save=savepath+savename[1])
show_image(img_sh_r, save=savepath+savename[2])
show_image(img_ro_p, save=savepath+savename[3])
show_image(img_ro_n, save=savepath+savename[4])
show_image(img_en_2, save=savepath+savename[5])

blob_construction(img_base, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=savepath+'Blob - '+savename[0])
blob_construction(img_sh_l, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=savepath+'Blob - '+savename[1])
blob_construction(img_sh_r, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=savepath+'Blob - '+savename[2])
blob_construction(img_ro_p, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=savepath+'Blob - '+savename[3])
blob_construction(img_ro_n, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=savepath+'Blob - '+savename[4])
blob_construction(img_en_2, size=3, threshold=0.02, initial=2, factor=1.25, level=15, sigma=1, save=savepath+'Blob - '+savename[5])
