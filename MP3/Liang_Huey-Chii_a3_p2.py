import os
# import sys
import glob
# import re
import time
# import matplotlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#####################################
### Provided functions start here ###
#####################################

## Image loading and saving
def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.  
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """
    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs

def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)

## Plot the height map
def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def display_albedo(albedo_image):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    
def display_height_map(albedo_image, height_map, viewpoint):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(viewpoint[0], viewpoint[1])
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)

## Plot the surface normals
def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])

#######################################
### Your implementation starts here ###
#######################################

def preprocess(ambimage, imarray):
    """
    preprocess the data: 
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """
    processed_imarray = imarray - np.stack([ambient_image]*imarray.shape[2], axis=2)
    processed_imarray[processed_imarray < 0] = 0
    processed_imarray = processed_imarray / 255
    return processed_imarray

def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """
    h, w, N = imarray.shape[0], imarray.shape[1], imarray.shape[2]
    imarray_new = imarray.reshape(h * w, N).transpose()
    g = np.linalg.lstsq(light_dirs, imarray_new, rcond=None)[0]
    albedo_image = np.linalg.norm(g, axis=0).reshape(h, w)
    surface_normals = g.transpose().reshape(h, w, 3) / np.stack([albedo_image]*3, axis=2)
    return albedo_image, surface_normals

def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals: h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    fx = surface_normals[:,:,0] / surface_normals[:,:,2]
    fy = surface_normals[:,:,1] / surface_normals[:,:,2]
    
    def average():
        height_map_rc = np.vstack([np.cumsum(fx, axis=1)[0,:]] * fx.shape[0]) + np.cumsum(fy, axis=0)
        height_map_cr = np.vstack([np.cumsum(fy, axis=0)[:,0]] * fy.shape[1]).transpose() + np.cumsum(fx, axis=1)
        height_map = (height_map_rc + height_map_cr) / 2
        return height_map
    
    def row():
        height_map = np.vstack([np.cumsum(fx, axis=1)[0,:]] * fx.shape[0]) + np.cumsum(fy, axis=0)
        return height_map
    
    def column():
        height_map = np.vstack([np.cumsum(fy, axis=0)[:,0]] * fy.shape[1]).transpose() + np.cumsum(fx, axis=1)
        return height_map
    
    def random():
        start = time.time()
        path_num = 10
        height_map = np.zeros((surface_normals.shape[0], surface_normals.shape[1]))
        for i in range(path_num):
            for y in range(height_map.shape[0]):
                for x in range(height_map.shape[1]):
                    h, w, fxy = 0, 0, 0
                    while (h <= y and w <= x):
                        if (h == y and w == x):
                            break
                        elif h == y:
                            w = w + 1
                            fxy = fxy + fx[h,w]
                        elif w == x:
                            h = h + 1
                            fxy = fxy + fy[h,w]
                        else:
                            if np.random.randint(2):
                                w = w + 1
                                fxy = fxy + fx[h,w]
                            else:
                                h = h + 1
                                fxy = fxy + fy[h,w]
                    height_map[y,x] = height_map[y,x] + fxy
        height_map = height_map / path_num
        end = time.time()
        print('Integration method: random')
        print('Execution time:', end - start)
        return height_map
    
    method = {'average': average, 'column': column, 'row': row, 'random': random}
    height_map = method[integration_method]()
    return height_map

## Main function
if __name__ == '__main__':
    root_path = 'C:/Users/liang/Desktop/croppedyale/'
    
    subject_name = 'yaleB01'
    # subject_name = 'yaleB02'
    # subject_name = 'yaleB05'
    # subject_name = 'yaleB07'
    
    integration_method = 'average'
    # integration_method = 'column'
    # integration_method = 'row'
    # integration_method = 'random'
    
    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name, 64)
    processed_imarray = preprocess(ambient_image, imarray)
    albedo_image, surface_normals = photometric_stereo(processed_imarray, light_dirs)
    height_map = get_surface(surface_normals, integration_method)

    # save_flag = True
    save_flag = False
    
    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)
    display_albedo(albedo_image)
    display_height_map(albedo_image, height_map, viewpoint=[20, 20]) # front
    display_height_map(albedo_image, height_map, viewpoint=[20, 70]) # side
    display_height_map(albedo_image, height_map, viewpoint=[60, 20]) # top
    
    # Calculate average execution time
    if integration_method != 'random':
        start = time.time()
        for i in range(10):
            albedo_image, surface_normals = photometric_stereo(processed_imarray, light_dirs)
        end = time.time()
        print('Integration method:', integration_method)
        print('Average execution time:', (end - start)/10)

#%%
# Show iimages from imarray
# for i in range(imarray.shape[2]):
#     plt.figure(figsize=(20,20))
#     plt.imshow(imarray[:,:,i], cmap='gray')
#     plt.title(i, fontsize=40)
#     plt.axis('off')

# imarray = np.delete(imarray, [1, 11, 14, 38], axis=2)
# light_dirs = np.delete(light_dirs, [1, 11, 14, 38], axis=0)
