# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    B = None
    L = None

    u, sv, vh = np.linalg.svd(I, full_matrices=False)
    sv[3:] = 0

    L = (u*sv)[:, :3].T
    B = vh[:3,:]

    return B, L

if __name__ == "__main__":

    I, realL, s = I, L, s = loadData()

    B, L = estimatePseudonormalsUncalibrated(I)

    print(L)
    print(realL)

    _, normalsBad = estimateAlbedosNormals(B)

    BI = enforceIntegrability(B, s)

    # albedos, normals = estimateAlbedosNormals(BI)

    # displayAlbedosNormals(albedos, normals, s)

    # surf1 = estimateShape(normalsBad, s)
    # surf2 = estimateShape(normals, s)
    # plotSurface(np.clip(surf1, -1000, 1000))
    # plotSurface(surf2)

    G = np.eye(3)
    G[2,2] = 0
    G[2,:] = [10,0,.7]
    Ginv = np.linalg.inv(G)

    Bnew  = Ginv.T @ BI
    _, normals = estimateAlbedosNormals(Bnew)
    surf = estimateShape(normals, s)
    plotSurface(surf)