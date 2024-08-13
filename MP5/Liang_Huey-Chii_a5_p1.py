# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 03:14:49 2023
@author: liang
"""
# %matplotlib widget
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%%
## Load the data matrix and normalize the point coordinates by translating them to the mean of the 
## points in each view (see lecture for details).
data = []
f = open('./factorization_data/measurement_matrix.txt', 'r')
for line in f:
    data.append(np.array(line.split(' '), dtype=float))
D = np.array(data) - np.mean(data, axis=1)[:, np.newaxis]

#%%
## Apply SVD to the 2M x N data matrix to express it as D = U @ W @ V' (using NumPy notation) where 
## U is a 2Mx3 matrix, W is a 3x3 matrix of the top three singular values, and V is a Nx3 matrix. 
## You can use numpy.linalg.svd to compute this decomposition. Next, derive structure and motion 
## matrices from the SVD as explained in the lecture.
U, s, V = np.linalg.svd(D)
U3 = U[:,:3]
s3 = np.identity(3) * s[:3]
V3 = V[:3,:]
Motion = U3 @ np.sqrt(s3)
Structure = np.sqrt(s3) @ V3

#%%
## Find the matrix Q to eliminate the affine ambiguity using the method described on slide 32.
F = Motion.shape[0] // 2
i, j = np.zeros((F, Motion.shape[1])), np.zeros((F, Motion.shape[1]))
for idx in range(F):
    idx_i = 2 * idx
    idx_j = 2 * idx + 1
    i[idx,:] = Motion[idx_i,:]
    j[idx,:] = Motion[idx_j,:]

A = np.zeros((2*F, 6))
for f in range(F):
    A[0*F+f, 0] = i[f,0] **2 - j[f,0] **2
    A[0*F+f, 1] = i[f,1] **2 - j[f,1] **2
    A[0*F+f, 2] = i[f,2] **2 - j[f,2] **2
    A[0*F+f, 3] = 2 * i[f,0] * i[f,1] - 2 * j[f,0] * j[f,1]
    A[0*F+f, 4] = 2 * i[f,0] * i[f,2] - 2 * j[f,0] * j[f,2]
    A[0*F+f, 5] = 2 * i[f,1] * i[f,2] - 2 * j[f,1] * j[f,2]
    A[1*F+f, 0] = i[f,0] * j[f,0]
    A[1*F+f, 1] = i[f,1] * j[f,1]
    A[1*F+f, 2] = i[f,2] * j[f,2]
    A[1*F+f, 3] = i[f,0] * j[f,1] + i[f,1] * i[f,0]
    A[1*F+f, 4] = i[f,0] * j[f,2] + i[f,2] * j[f,0]
    A[1*F+f, 5] = i[f,1] * j[f,2] + i[f,2] * j[f,1]

_, _, v = np.linalg.svd(A)
x = v[-1,:]
L = np.array([[x[0], x[3], x[4]], 
              [x[3], x[1], x[5]], 
              [x[4], x[5], x[2]]])

Q = np.linalg.cholesky(L)
Motion_new = Motion @ Q
Structure_new = np.linalg.inv(Q) @ Structure

#%%
## Use matplotlib to display the 3D structure (in your report, you may want to include snapshots 
## from several viewpoints to show the structure clearly). Discuss whether or not the reconstruction 
## has an ambiguity.
def plot_3D(structure, view=[30,-60]):
    plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')
    ax.scatter3D(Structure[0,:], Structure[1,:], Structure[2,:], c='r')
    ax.view_init(view[0],view[1])
    plt.title('3D structure')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return

plot_3D(Structure_new, view=[-160,-15])
plot_3D(Structure_new, view=[-160,-30])
plot_3D(Structure_new, view=[-160,-45])

#%%
## Display three frames with both the observed feature points and the estimated projected 3D points 
## overlayed. Report your total residual (sum of squared Euclidean distances, in pixels, between the 
## observed and the reprojected features) over all the frames, and plot the per-frame residual as a 
## function of the frame number.
filelist = []
for file in os.listdir('./factorization_data/'):
    filelist.append('./factorization_data/' + file)

residual = []
for i in range(101):
    feat_pts = np.array(data)[2*i:2*i+2,:].astype(np.int16)
    proj_mtx = Motion_new[2*i:2*i+2,:]
    proj_pts = (proj_mtx @ Structure_new + np.mean(feat_pts, axis=1)[:, np.newaxis]).astype(np.int16)
    residual.append(np.mean((feat_pts[0,:] - proj_pts[0,:])**2 + (feat_pts[1,:] - proj_pts[1,:])**2))
    
    img = cv2.imread(filelist[i])
    fig = img.copy()
    for j in range(feat_pts.shape[1]):
        fig = cv2.circle(fig, (feat_pts[0,j], feat_pts[1,j]), radius=4, color=[0,255,0], thickness=-1)
        fig = cv2.circle(fig, (proj_pts[0,j], proj_pts[1,j]), radius=4, color=[255,0,0], thickness=-1)
    # cv2.imwrite('./results/frame_' + str(i+1) + '.jpg', fig)
    # cv2.imshow('frame' + str(i+1), fig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

residual_tot = np.sum(residual)
print('Total residual is', residual_tot)
plt.figure(figsize=(8,6))
plt.plot(np.arange(1,102), np.array(residual))
plt.title('Per-frame Residual')
plt.xlabel('frame number')
plt.ylabel('residual')
plt.grid()
