# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """
    #assuming res is [width, height]
    width, height = res

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    X = pxSize * (X - width/2)
    Y = -pxSize * (Y - height/2)

    dists = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    Z = np.where(dists <= rad, np.sqrt(np.clip(rad**2 - dists**2,0,None)), 0)

    diffs = np.dstack((X, Y, Z)) - center.reshape((1,1,3))
    norms = np.linalg.norm(diffs, axis = 2)
    norms = np.dstack((norms, norms, norms))
    ns = diffs / norms
    image = np.where(dists <= rad, np.clip(np.dot(ns, light), 0, None), 0)

    return image

def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    im = [str(x) for x in range(1,8)]
    I = []
    for i in im:
        arr = imread(path + 'input_{}.tif'.format(i)).astype(np.uint16)
        arr = rgb2xyz(arr)[:,:,1]
        s = arr.shape
        I.append(arr.flatten())

    I = np.array(I)

    L = np.load(path + 'sources.npy').T

    _, sv, _ = np.linalg.svd(I, full_matrices=False)
    print("I matrix singular values:\n\t{}".format(sv))

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    S = L.T

    #moore pensore pseudoinverse
    B = np.linalg.inv(S.T @ S) @ S.T @ I

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    #albedo
    albedos = np.linalg.norm(B, axis = 0)

    #unit normals
    normals = B / albedos

    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns\
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    #albedo
    albedoIm = albedos.reshape(s)
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    #unit normals
    normalIm = normals.T.reshape((s[0],s[1],3))
    plt.imshow((1 + normalIm)/2, cmap='rainbow')
    plt.show()

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    zx = (-normals[0,:]/normals[2,:]).reshape(s)
    zy = (-normals[1,:]/normals[2,:]).reshape(s)

    surface = integrateFrankot(zx, zy)

    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    f = plt.figure()
    ax = f.gca(projection='3d')
    x = np.arange(surface.shape[1])
    y = np.arange(surface.shape[0])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, -surface, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    plt.show()

if __name__ == '__main__':

    I, L, s = loadData()
    B = estimatePseudonormalsCalibrated(I, L)

    albedos, normals = estimateAlbedosNormals(B)
    albedosIm, normalsIm = displayAlbedosNormals(albedos, normals, s)

    surface = estimateShape(normals, s)
    plotSurface(surface)
    #Question 1b
    if True:
        sphere1 = renderNDotLSphere(np.array([0, 0, 0]), .0075, 
            np.array([ 1, 1,1])/np.sqrt(3), 7e-6, [3840, 2160])
        sphere2 = renderNDotLSphere(np.array([0, 0, 0]), .0075, 
            np.array([ 1,-1,1])/np.sqrt(3), 7e-6, [3840, 2160])
        sphere3 = renderNDotLSphere(np.array([0, 0, 0]), .0075, 
            np.array([-1,-1,1])/np.sqrt(3), 7e-6, [3840, 2160])
        plt.imshow(sphere1, cmap='gray')
        plt.show()
        plt.imshow(sphere2, cmap='gray')
        plt.show()
        plt.imshow(sphere3, cmap='gray')
        plt.show()

    #question 1h
    if True:
        g = np.array(range(1,17)).reshape((4,4))
        gx = np.diff(g, axis = 1)
        gy = np.diff(g, axis = 0)
        g11 = 1
        g1r = np.hstack((g11, g11 + np.cumsum(gx, axis = 1)[0,:]))
        g1 = np.vstack((g1r, g1r + np.cumsum(gy, axis = 0)))
        print("First row, then cols:\n{}".format(g1))

        g2c = np.vstack((g11, g11 + np.cumsum(gy, axis = 0)[:,0].reshape(3,1)))
        g2 = np.hstack((g2c, g2c + np.cumsum(gx, axis = 1)))
        print("First col, then rows:\n{}".format(g2))