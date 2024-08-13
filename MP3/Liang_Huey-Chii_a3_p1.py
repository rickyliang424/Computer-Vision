import cv2
import scipy 
import skimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 2. Load both images, convert to double and to grayscale.
path = 'C:/Users/liang/Desktop/'
img_left = np.array(Image.open(path+'left.jpg').convert("L"))
img_right = np.array(Image.open(path+'right.jpg').convert("L"))

#%%
## 4. Extract keypoints and compute descriptors.
kp_left, des_left = cv2.SIFT_create().detectAndCompute(img_left, None)
kp_right, des_right = cv2.SIFT_create().detectAndCompute(img_right, None)
sift_left = cv2.drawKeypoints(img_left, kp_left, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_right = cv2.drawKeypoints(img_right, kp_right, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Image.fromarray(sift_left).show()
# Image.fromarray(sift_right).show()
# Image.fromarray(sift_left).save(path + 'sift_left.jpg')
# Image.fromarray(sift_right).save(path + 'sift_right.jpg')

#%%
## 5. Compute distances between every descriptor in one image and every descriptor in the other image.
des_dist = scipy.spatial.distance.cdist(des_left, des_right, 'sqeuclidean')

#%%
## 6. Select putative matches based on the matrix of pairwise descriptor distances obtained above.
pair_num = 200
des_dist_idx = np.unravel_index(np.argsort(des_dist, axis=None), des_dist.shape)
idx_left = des_dist_idx[0][:pair_num]
idx_right = des_dist_idx[1][:pair_num]

kp_left_new, kp_right_new = [], []
for i in range(pair_num):
    kp_left_new.append(kp_left[idx_left[i]])
    kp_right_new.append(kp_right[idx_right[i]])
sift_left_new = cv2.drawKeypoints(img_left, kp_left_new, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_right_new = cv2.drawKeypoints(img_right, kp_right_new, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Image.fromarray(sift_left_new).show()
# Image.fromarray(sift_right_new).show()
# Image.fromarray(sift_left_new).save(path + 'sift_left_new.jpg')
# Image.fromarray(sift_right_new).save(path + 'sift_right_new.jpg')

#%%
## 7. Implement RANSAC to estimate a homography mapping one image onto the other.
init_pts = 4
threshold = 5
iter_num = 10000

H_list, inliers_list, inlier_num_list, residual_list = [], [], [], []
for i in range(iter_num):
    Lx, Ly, Rx, Ry = [], [], [], []
    rand_idx = np.random.randint(0, pair_num, init_pts)
    for i in rand_idx:
        Lx.append(kp_left_new[i].pt[0])
        Ly.append(kp_left_new[i].pt[1])
        Rx.append(kp_right_new[i].pt[0])
        Ry.append(kp_right_new[i].pt[1])
    
    A = np.array([
        [-Lx[0], -Ly[0], -1, 0, 0, 0, Lx[0]*Rx[0], Ly[0]*Rx[0], Rx[0]], 
        [0, 0, 0, -Lx[0], -Ly[0], -1, Lx[0]*Ry[0], Ly[0]*Ry[0], Ry[0]], 
        [-Lx[1], -Ly[1], -1, 0, 0, 0, Lx[1]*Rx[1], Ly[1]*Rx[1], Rx[1]], 
        [0, 0, 0, -Lx[1], -Ly[1], -1, Lx[1]*Ry[1], Ly[1]*Ry[1], Ry[1]], 
        [-Lx[2], -Ly[2], -1, 0, 0, 0, Lx[2]*Rx[2], Ly[2]*Rx[2], Rx[2]], 
        [0, 0, 0, -Lx[2], -Ly[2], -1, Lx[2]*Ry[2], Ly[2]*Ry[2], Ry[2]], 
        [-Lx[3], -Ly[3], -1, 0, 0, 0, Lx[3]*Rx[3], Ly[3]*Rx[3], Rx[3]], 
        [0, 0, 0, -Lx[3], -Ly[3], -1, Lx[3]*Ry[3], Ly[3]*Ry[3], Ry[3]]])
    U, s, V = np.linalg.svd(A)
    H  = V[len(V)-1].reshape((3,3))
    
    inlier_num, residual_tot, inlier_idx_list = 0, 0, []
    for i in range(pair_num):
        Lx = kp_left_new[i].pt[0]
        Ly = kp_left_new[i].pt[1]
        Rx = kp_right_new[i].pt[0]
        Ry = kp_right_new[i].pt[1]
        Lx_new = (H[0,0]*Lx + H[0,1]*Ly + H[0,2]) / (H[2,0]*Lx + H[2,1]*Ly + H[2,2])
        Ly_new = (H[1,0]*Lx + H[1,1]*Ly + H[1,2]) / (H[2,0]*Lx + H[2,1]*Ly + H[2,2])
        residual = (Lx_new - Rx)**2 + (Ly_new - Ry)**2
        if residual <= threshold:
            inlier_num = inlier_num + 1
            residual_tot = residual_tot + residual
            inlier_idx_list.append(i)
            
    H_list.append(H)
    inliers_list.append(inlier_idx_list)
    inlier_num_list.append(inlier_num)
    residual_list.append(residual_tot)

optimal_idx = np.argmax(inlier_num_list)
optimal_inliers = inliers_list[optimal_idx]
optimal_H = H_list[optimal_idx]
print("The number of inliers is:", inlier_num_list[optimal_idx])
print("The average residual for the inliers is:", residual_list[optimal_idx] / inlier_num_list[optimal_idx])

inliers = []
for i in optimal_inliers:
    Lx = kp_left_new[i].pt[0]
    Ly = kp_left_new[i].pt[1]
    Rx = kp_right_new[i].pt[0]
    Ry = kp_right_new[i].pt[1]
    inliers.append([Lx, Ly, Rx, Ry])
inliers = np.array(inliers)

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    ax.set_aspect('equal')
    ax.imshow(np.hstack([img1, img2]), cmap='gray')
    ax.plot(inliers[:,0], inliers[:,1], '+r', markersize=10)
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r', markersize=10)
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]], [inliers[:,1], inliers[:,3]], 'r', linewidth=1)
    ax.axis('off')

fig, ax = plt.subplots(figsize=(40,20))
plot_inlier_matches(ax, img_left, img_right, inliers)

#%%
## Final optimal H and inliers
# optimal_H = np.array([[-1.94685535e-03,  1.86808028e-04,  9.56136542e-01], 
#                       [-3.75987919e-04, -1.82927626e-03,  2.92907978e-01], 
#                       [-1.21972495e-06, -3.46650116e-08, -7.19260832e-04]])
# optimal_inliers = [  0,   2,   3,   4,   9,  13,  14,  18,  19,  20, 
#                     21,  22,  26,  27,  28,  29,  30,  32,  33,  35, 
#                     36,  41,  44,  46,  49,  50,  51,  52,  53,  54, 
#                     58,  60,  62,  63,  64,  68,  69,  76,  77,  82, 
#                     85,  86,  87,  89,  90,  92,  93,  98, 103, 106, 
#                    108, 110, 111, 113, 115, 118, 122, 129, 131, 134, 
#                    135, 137, 155, 158, 163, 166, 170, 171, 173, 182, 
#                    183, 184]

#%%
## 8. Warp one image onto the other using the estimated transformation.
def warp_images(img_L, img_R, transform_matrix):
    corners = np.array([[0,0], [0,img_L.shape[0]], [img_L.shape[1],0], [img_L.shape[1],img_L.shape[0]]])
    corners_new = skimage.transform.ProjectiveTransform(transform_matrix)(corners)
    corner_min = np.min(np.vstack((corners, corners_new)), axis=0)
    corner_max = np.max(np.vstack((corners, corners_new)), axis=0)
    output_shape = np.ceil((corner_max - corner_min)[::-1]).astype(int)
    
    transform = skimage.transform.ProjectiveTransform(optimal_H)
    offset = skimage.transform.SimilarityTransform(translation = -corner_min)
    inverse_map = (transform + offset).inverse
    img_L_new_0 = skimage.transform.warp(img_L, inverse_map, output_shape=output_shape, cval=0)
    img_L_new_1 = skimage.transform.warp(img_L, inverse_map, output_shape=output_shape, cval=-1)
    img_R_new = skimage.transform.warp(img_R, offset.inverse, output_shape=output_shape, cval=0)
    
    if len(img_L.shape) == 3:
        img_new = np.zeros(np.hstack((output_shape,3)))
        img_new = img_new + img_R_new
        img_L_new_1_sum = np.sum(img_L_new_1, axis=2)
        for i in range(len(img_L.shape)):
            img_new_temp = img_new[:,:,i]
            img_new_temp[img_L_new_1_sum > 0] = 0
            img_L_new_0_temp = img_L_new_0[:,:,i]
            img_L_new_0_temp[img_new_temp > 0] = 0
            img_new[:,:,i] = img_new_temp + img_L_new_0_temp
        plt.figure(figsize = (25, 10))
        plt.imshow(img_new)
        plt.axis('off')
    else:
        img_new = np.zeros(output_shape)
        img_new = img_new + img_R_new
        img_new[img_L_new_1 > 0] = 0
        img_L_new_0[img_new > 0] = 0
        img_new = img_new + img_L_new_0
        plt.figure(figsize = (25, 10))
        plt.imshow(img_new, cmap='gray')
        plt.axis('off')
    return img_new

#%%
## 9. Create a new image big enough to hold the panorama and composite the two images into it.
img_new = warp_images(img_left, img_right, optimal_H)

#%%
## 10. Create a color panorama by applying the same compositing step to each of the color channels separately.
img_left_RGB = np.array(Image.open(path+'left.jpg'))
img_right_RGB = np.array(Image.open(path+'right.jpg'))
img_new_RGB = warp_images(img_left_RGB, img_right_RGB, optimal_H)
