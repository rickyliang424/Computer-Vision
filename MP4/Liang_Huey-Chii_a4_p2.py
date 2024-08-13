## Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#%% Fundamental Matrix Estimation - 1
## load images and match files for the first example
I1 = Image.open('C:/Users/liang/Desktop/MP4_part2_data/library1.jpg')
I2 = Image.open('C:/Users/liang/Desktop/MP4_part2_data/library2.jpg')
matches = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/library_matches.txt')
# I1 = Image.open('C:/Users/liang/Desktop/MP4_part2_data/lab1.jpg')
# I2 = Image.open('C:/Users/liang/Desktop/MP4_part2_data/lab2.jpg')
# matches = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/lab_matches.txt')

## this is a N x 4 file where the first two numbers of each row
## are coordinates of corners in the first image and the last two
## are coordinates of corresponding corners in the second image: 
## matches(i,1:2) is a point in the first image
## matches(i,3:4) is a corresponding point in the second image
N = len(matches)

#%% Fundamental Matrix Estimation - 2
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
I3[:,:I1.size[0],:] = I1;
I3[:,I1.size[0]:,:] = I2;
fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(float) / 255)
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.plot(matches[:,2]+I1.size[0],matches[:,3], '+r')
ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], '--r')
plt.show()

#%% Fundamental Matrix Estimation - 3
## display second image with epipolar lines reprojected from the first image
def normalization(matches):
    N = len(matches)
    matches_mean = matches - np.mean(matches, axis=0)
    scale1 = np.sqrt(2 * N / np.sum(np.square(matches_mean[:,0:2])))
    scale2 = np.sqrt(2 * N / np.sum(np.square(matches_mean[:,2:4])))
    matches_norm = np.hstack((scale1 * matches_mean[:,0:2], scale2 * matches_mean[:,2:4]))
    T1 = np.array([[scale1, 0, -scale1 * np.mean(matches[:,0])], 
                    [0, scale1, -scale1 * np.mean(matches[:,1])], 
                    [0, 0, 1]])
    T2 = np.array([[scale2, 0, -scale2 * np.mean(matches[:,2])], 
                    [0, scale2, -scale2 * np.mean(matches[:,3])], 
                    [0, 0, 1]])
    return matches_norm, T1, T2

def fit_fundamental(matches, normalize=True):
    if normalize == True:
        matches, T1, T2 = normalization(matches)
    ## The eight point algorithm
    U_8p = np.zeros((N, 9)) 
    for i in range(N):
        x1, y1, x2, y2 = matches[i,:]
        U_8p[i,:] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    U, s, V = np.linalg.svd(U_8p)
    F_init = V[len(V)-1].reshape(3,3)
    ## enforce rank-2 constraint
    U, s, V = np.linalg.svd(F_init)
    s[-1] = 0
    s_new = s * np.identity(3)
    F = U @ s_new @ V
    F = np.matmul(np.matmul(U, s_new), V)
    if normalize == True:
        F = np.matmul(np.matmul(T2.T, F), T1)
    return F

## first, fit fundamental matrix to the matches
## this is a function that you should write
F = fit_fundamental(matches, normalize=True)
# F = fit_fundamental(matches, normalize=False)
M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
## transform points from the first image to get epipolar lines in the second image
L1 = np.matmul(F, M).transpose()

## find points on epipolar lines L closest to matches(:,3:4)
l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
L = np.divide(L1, np.kron(np.ones((3,1)), l).transpose())  # rescale the line
pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
closest_pt = matches[:,2:4] - np.multiply(L[:,0:2], np.kron(np.ones((2,1)), pt_line_dist).transpose())

## find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - np.c_[L[:,1], -L[:,0]] * 10  # offset from the closest point is 10 pixels
pt2 = closest_pt + np.c_[L[:,1], -L[:,0]] * 10

## display points and segments of corresponding epipolar lines
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(float) / 255)
ax.plot(matches[:,2], matches[:,3], '+r')
ax.plot([matches[:,2], closest_pt[:,0]], [matches[:,3], closest_pt[:,1]], 'r')
ax.plot([pt1[:,0], pt2[:,0]], [pt1[:,1], pt2[:,1]], 'g')
plt.show()

print('Residual =', np.mean(np.abs(pt_line_dist)))

#%% Camera Calibration
def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def find_proj_mtx(pts_2d, pts_3d):
    A = np.zeros((len(pts_3d), 12))
    for i in range(int(len(pts_3d) / 2)):
        A[i*2, 4:8] = np.array([pts_3d[i,0], pts_3d[i,1], pts_3d[i,2], 1])
        A[i*2, 8:12] = -pts_2d[i,1] * np.array([pts_3d[i,0], pts_3d[i,1], pts_3d[i,2], 1])
        A[i*2+1, 0:4] = np.array([pts_3d[i,0], pts_3d[i,1], pts_3d[i,2], 1])
        A[i*2+1, 8:12] = -pts_2d[i,0] * np.array([pts_3d[i,0], pts_3d[i,1], pts_3d[i,2], 1])
    U, s, V = np.linalg.svd(A)
    P = V[len(V)-1].reshape((3,4))
    return P

lab_matches = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/lab_matches.txt')
lab_3d_pts = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/lab_3d.txt')

lab1_proj_mtx = find_proj_mtx(matches[:,0:2], lab_3d_pts)
lab2_proj_mtx = find_proj_mtx(matches[:,2:4], lab_3d_pts)
proj1, lab1_residual = evaluate_points(lab1_proj_mtx, matches[:,0:2], lab_3d_pts)
proj2, lab2_residual = evaluate_points(lab2_proj_mtx, matches[:,2:4], lab_3d_pts)
print('Estimated camera projection matrix for lab1:\n', lab1_proj_mtx, '\n')
print('Estimated camera projection matrix for lab2:\n', lab2_proj_mtx, '\n')
print('The residual between the projected and observed 2D points for lab1:', lab1_residual)
print('The residual between the projected and observed 2D points for lab2:', lab2_residual)

#%% Camera Centers
library1_proj_mtx = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/library1_camera.txt')
library2_proj_mtx = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/library2_camera.txt')

def find_camera_center(proj_mtx):
    U, s, V = np.linalg.svd(proj_mtx)
    center = V[len(V)-1]
    center = center / center[-1]
    return center

lab1_center = find_camera_center(lab1_proj_mtx)
lab2_center = find_camera_center(lab2_proj_mtx)
library1_center = find_camera_center(library1_proj_mtx)
library2_center = find_camera_center(library2_proj_mtx)
print(lab1_center)
print(lab2_center)
print(library1_center)
print(library2_center)

#%% Triangulation
lab_matches = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/lab_matches.txt')
N = len(lab_matches)

attach = np.ones((N, 1))
x1 = np.hstack((lab_matches[:, : 2], attach))
x2 = np.hstack((lab_matches[:, 2 :], attach))

X_3d = np.zeros((N, 4))
for i in range(N):
    x_1 = np.zeros((3, 3))
    x_1[0, 1] = -1 * x1[i, 2]
    x_1[0, 2] = x1[i, 1]
    x_1[1, 0] = x1[i, 2]
    x_1[1, 2] = -1 * x1[i, 0]
    x_1[2, 0] = -1 * x1[i, 1]
    x_1[2, 1] = x1[i, 0]
    
    x_2 = np.zeros((3, 3))
    x_2[0, 1] = -1 * x2[i, 2]
    x_2[0, 2] = x2[i, 1]
    x_2[1, 0] = x2[i, 2]
    x_2[1, 2] = -1 * x2[i, 0]
    x_2[2, 0] = -1 * x2[i, 1]
    x_2[2, 1] = x2[i, 0]

    A_x1 = np.matmul(x_1, lab1_proj_mtx)
    A_x2 = np.matmul(x_2, lab2_proj_mtx)
    A = np.vstack((A_x1, A_x2))

    U, s, V = np.linalg.svd(A)
    sol = V[len(V)-1]
    X_3d[i] = sol / sol[-1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lab1_center[0], lab1_center[1], lab1_center[2])
ax.scatter(lab2_center[0], lab2_center[1], lab2_center[2])
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])
ax.view_init(40, 60)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#%%
library_matches = np.loadtxt('C:/Users/liang/Desktop/MP4_part2_data/library_matches.txt')
N = len(library_matches)

attach = np.ones((N, 1))
x1 = np.hstack((library_matches[:, : 2], attach))
x2 = np.hstack((library_matches[:, 2 :], attach))

X_3d = np.zeros((N, 4))
for i in range(N):
    x_1 = np.zeros((3, 3))
    x_1[0, 1] = -1 * x1[i, 2]
    x_1[0, 2] = x1[i, 1]
    x_1[1, 0] = x1[i, 2]
    x_1[1, 2] = -1 * x1[i, 0]
    x_1[2, 0] = -1 * x1[i, 1]
    x_1[2, 1] = x1[i, 0]
    
    x_2 = np.zeros((3, 3))
    x_2[0, 1] = -1 * x2[i, 2]
    x_2[0, 2] = x2[i, 1]
    x_2[1, 0] = x2[i, 2]
    x_2[1, 2] = -1 * x2[i, 0]
    x_2[2, 0] = -1 * x2[i, 1]
    x_2[2, 1] = x2[i, 0]

    A_x1 = np.matmul(x_1, library1_proj_mtx)
    A_x2 = np.matmul(x_2, library2_proj_mtx)
    A = np.vstack((A_x1, A_x2))

    U, s, V = np.linalg.svd(A)
    sol = V[len(V)-1]
    X_3d[i] = sol / sol[-1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(library1_center[0], library1_center[1], library1_center[2])
ax.scatter(library2_center[0], library2_center[1], library2_center[2])
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])
ax.view_init(40, 60)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

