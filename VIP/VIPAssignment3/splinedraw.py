# -*- coding: utf-8 -*-
"""
Example code for getting a curve: points are selected and a cubic spline with 
user defined amount of samples is built and returned.
@author: francois
"""


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

              
def getInitialCurve(im, nbr_points):
    """ Select a number of vertices for builing a cubic spline planar 
    curve and return a set of nbr_points of that curve, with equidistant 
    parametrization.    
    """
    im = np.array(im)
    plt.gray()
    plt.imshow(np.flipud(im), origin='lower')
    vertices = np.array(plt.ginput(0, timeout=10))
    # I add an extra point to close the curve
    # this is for spline interpolation
    # I will remove it from the interpolant
    vx = vertices[:,0]
    vx = np.append(vx, vx[0])
    vy = vertices[:,1]
    vy = np.append(vy, vy[0])

    nbr_vertices = len(vx)
    t = np.linspace(0,nbr_vertices,nbr_vertices) #returns an array of numbers from 0 to nbr of clicks+1. Num of samples are also nbr of clicks+1. E.g. 4 clicks: ([0., x.x, x.x, x.x, 5.])
    sx = interpolate.splrep(t, vx, s= 0) 
    sy = interpolate.splrep(t, vy, s= 0)
    tnew = np.linspace(0, nbr_vertices, nbr_points+1)
    
    vxnew = interpolate.splev(tnew, sx, der=0)
    vynew = interpolate.splev(tnew, sy, der=0)
    # I don't want the last point
    return vxnew[:-1], vynew[:-1], vertices
    
    
def drawCurve(x,y, im=None):
    """ Draw the curve specified by coordinates x and y on top of the 
    image im. The curve is closed. """
    #if not im == None:
    #    im = np.array(im)
    #    plt.gray()
     #   plt.imshow(np.flipud(im), origin='lower')

    x = np.array(x)
    y = np.array(y)
    
    cx = np.append(x, x[0])
    cy = np.append(y, y[0])
    plt.axis([0, len(im[1]), 0, len(im)])
    plt.plot(cx, cy, '+-')
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    plt.show
     
    
    
if __name__ == "__main__":
    from resample import resample    
    im = np.zeros((300,300))
    im[120:180,120:180] = 1
    
    x,y,points = getInitialCurve(im, 30)
    drawCurve(x, y, im)
    plt.plot(points[:,0], points[:,1], 'r+')
    f = np.vstack((x,y)).T
    f = np.matrix(f)
    rf = np.array(resample(f))
    rfx = rf[:,0]
    rfy = rf[:,1]
    drawCurve(rfx, rfy)
    
    
        
    
    
