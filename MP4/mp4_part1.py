#!/usr/bin/env python
# coding: utf-8
# Part 3: Single-View Geometry

#%% Common imports
%matplotlib tk
import numpy as np
import sympy as sp
from PIL import Image
import matplotlib.pyplot as plt

#%% Provided functions
def get_input_lines(im, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.imshow(im)
    plt.show()
    print('Set at least %d lines to compute vanishing point' % min_lines)
    while True:
        print('Click the two endpoints, use the right key to undo, and use the middle key to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print('Need at least %d lines, you have %d now' % (min_lines, n))
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')
        n += 1
    return n, lines, centers

def plot_lines_and_vp(im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10

    plt.figure()
    plt.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    plt.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    plt.show()

def get_top_and_bottom_coordinates(im):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    plt.figure()
    plt.imshow(im)
    plt.show()

    print('Click on the top coordinate of the object')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of the object')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')
    return np.array([[x1, x2], [y1, y2], [1, 1]])

#%% Your implementation
def get_vanishing_point(lines):
    """
    Solves for the vanishing point using the user-input lines.
    """
    # <YOUR IMPLEMENTATION>
    pts_list = []
    for i in range(lines.shape[1]):
        for j in range(i+1, lines.shape[1]):
            pts_list.append(np.cross(lines[:, i], lines[:, j]))    
    pts_array = np.array(pts_list) / np.expand_dims(np.array(pts_list)[:,-1], axis=1)
    vpts = np.mean(pts_array, axis=0)
    return vpts

def get_horizon_line(vpts):
    """
    Calculates the ground horizon line.
    """
    # <YOUR IMPLEMENTATION>
    hline = np.cross(vpts[:,0], vpts[:,1])
    horizon_line = hline / np.hypot(hline[0], hline[1])
    return horizon_line

def plot_horizon_line(im, vpts):
    """
    Plots the horizon line.
    """
    # <YOUR IMPLEMENTATION>
    plt.figure()
    plt.imshow(im)
    plt.plot([vpts[0,0], vpts[0,1]], [vpts[1,0], vpts[1,1]], 'b')
    plt.plot(vpts[0,0], vpts[1,0], 'rx')
    plt.plot(vpts[0,1], vpts[1,1], 'rx')
    plt.show()
    return

def get_camera_parameters(vpts):
    """
    Computes the camera parameters. Hint: The SymPy package is suitable for this.
    """
    # <YOUR IMPLEMENTATION>
    f = sp.Symbol('f')
    u = sp.Symbol('u')
    v = sp.Symbol('v')
    v1 = sp.Matrix(vpts[:,0])
    v2 = sp.Matrix(vpts[:,1])
    v3 = sp.Matrix(vpts[:,2])
    K_inv = sp.Matrix([[f, 0, u], [0, f, v], [0, 0, 1]]).inv()
    e12 = v1.T * K_inv.T * K_inv * v2
    e23 = v2.T * K_inv.T * K_inv * v3
    e31 = v3.T * K_inv.T * K_inv * v1
    sol = sp.solve([e12, e23, e31], [f, u, v])
    if len(sol) == 0:
        f, u, v = None, None, None
        print('No solution... Please select new vanishing points!')
    else:
        f, u, v = sol[0]
    return f, u, v

def get_rotation_matrix(vpts, f, u, v):
    """
    Computes the rotation matrix using the camera parameters.
    """
    # <YOUR IMPLEMENTATION>
    vpts_x, vpts_y, vpts_z = vpts[:,1], vpts[:,2], vpts[:,0]
    K = np.array([[f, 0, u], [0, f, v], [0, 0, 1]]).astype(np.float64)
    K_inv = np.linalg.inv(K)
    r1 = K_inv @ vpts_x
    r2 = K_inv @ vpts_y
    r3 = K_inv @ vpts_z
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = r3 / np.linalg.norm(r3)
    R = np.stack((r1, r2, r3), axis=1)
    return R

def estimate_height(vpts, coords):
    """
    Estimates height for a specific object using the recorded coordinates. You might need to plot additional images here for
    your report.
    """
    # <YOUR IMPLEMENTATION>
    vx, vy, vz = vpts[:,0], vpts[:,1], vpts[:,2]
    t0, b0 = coords[0][:,0], coords[0][:,1]
    r, b = coords[1][:,0], coords[1][:,1]
    v = np.cross(np.cross(b, b0), np.cross(vx, vy))
    v = v / v[-1]
    t = np.cross(np.cross(v, t0), np.cross(r, b))
    t = t / t[-1]
    cross_ratio = (np.linalg.norm(t - b) * np.linalg.norm(vz - r)) / (np.linalg.norm(r - b) * np.linalg.norm(vz - t))
    return cross_ratio

def plot_obj_height(im, coords):
    plt.figure()
    plt.imshow(im)
    obj_1t, obj_1b = coords[0][:,0], coords[0][:,1]
    obj_2t, obj_2b = coords[1][:,0], coords[1][:,1]
    plt.plot([obj_1t[0], obj_1b[0]], [obj_1t[1], obj_1b[1]], 'b')
    plt.plot([obj_2t[0], obj_2b[0]], [obj_2t[1], obj_2b[1]], 'b')
    plt.plot(obj_1t[0], obj_1t[1], 'rx')
    plt.plot(obj_1b[0], obj_1b[1], 'rx')
    plt.plot(obj_2t[0], obj_2t[1], 'rx')
    plt.plot(obj_2b[0], obj_2b[1], 'rx')
    plt.show()
    return

#%% Main function
im = np.asarray(Image.open('C:/Users/liang/Desktop/CSL.jpg'))

# Part 1
# Get vanishing points for each of the directions
num_vpts = 3
vpts = np.zeros((3, num_vpts))
for i in range(num_vpts):
    print('Getting vanishing point %d' % i)
    # Get at least three lines from user input
    n, lines, centers = get_input_lines(im)
    # <YOUR IMPLEMENTATION> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(lines)
    # Plot the lines and the vanishing point
    plot_lines_and_vp(im, lines, vpts[:, i])
# <YOUR IMPLEMENTATION> Get the ground horizon line
horizon_line = get_horizon_line(vpts)
# <YOUR IMPLEMENTATION> Plot the ground horizon line
plot_horizon_line(im, vpts)

# Part 2
# <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
f, u, v = get_camera_parameters(vpts)

# Part 3
# <YOUR IMPLEMENTATION> Solve for the rotation matrix
R = get_rotation_matrix(vpts, f, u, v)

# Part 4
# Record image coordinates for each object and store in map
coords = []
for i in range(2):
    coord = get_top_and_bottom_coordinates(im)
    coords.append(coord)
# <YOUR IMPLEMENTATION> Estimate heights
plot_obj_height(im, coords)
cross_ratio = estimate_height(vpts, coords)
print('The cross-ratio of the two gable is ', round(cross_ratio, 2))
