# -*- coding: utf-8 -*-
 
from PIL import Image
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.interpolate import *
from scipy import integrate
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as pim
import math
import interpimage
import time
from matplotlib.widgets import Button
 
 
sigma = 5
im1=np.array(Image.open('images/blacksquare.png'), dtype=float)
gauss=gf(im1,sigma,2)
 
alpha = 0.51
beta = 1
gamma = 5
ts = 100
points = 500
thrs = 30
 
#Apply gaussian
Ix=gf(im1,sigma,(1,0))
Iy=gf(im1,sigma,(0,1))
 
Ixx=gf(im1,sigma,(2,0))
Iyy=gf(im1,sigma,(0,2))
Ixy=gf(im1,sigma,(1,1))
 
F = -(Ix**2 + Iy**2)
 
Fx = -2*(Ix*Ixx + Iy*Ixy)
Fy = -2*(Ix*Ixy + Iy*Iyy)
 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_aspect("equal")
#ax=plt.imshow(F, cmap=plt.cm.Greys_r)
 
xy=plt.ginput(n=0, timeout=30, show_clicks=True, mouse_add=1, mouse_pop=2, mouse_stop=3) #xy is a list of plots
 
 
#Load initial plots into separate variables and create array
x = [p[0] for p in xy]
y = [p[1] for p in xy]
 
x=np.array(x)
y=np.array(y)
 
#Add first element as last element too
xnew=np.hstack((x,x[0]))
ynew=np.hstack((y,y[0]))
 
#Perform spline representation and evaluate to create ellipse
tck,u=splprep([xnew,ynew],s=0)
unew=np.arange(0,1,0.002)
 
out=splev(unew,tck)
 
#Draw ellipse
#plt.figure()
#plt.imshow(F, cmap=plt.cm.Greys_r)
#plt.plot(out[0],out[1])
r, = ax1.plot(out[0],out[1],'b')
 
#Load all ellipse data points into x and y vars
x,y=out[0],out[1]
 
curveE=[]
bendE=[]
imageE=[]
#Calculate curve and bending energy terms
for p in range(len(x)-1):
    if x[p+1] < len(x):
        curveE.append(((x[p+1]-x[p])**2)+(y[p+1]-y[p])**2)
        bendE.append(math.sqrt(((((y[p+1]-y[p])/(x[p+1]-x[p])))-(((y[p]-y[p-1])/(x[p]-x[p-1]))))**2))
        imageE.append(F[y[p],x[p]])
 
curveE=sum(curveE)
bendE=sum(bendE)
imageE=sum(imageE)
snakeE = curveE+bendE+imageE
 
print curveE, bendE,imageE
print snakeE
 
M = np.zeros((1,points))
M[0,0] = 1+ts*(2*alpha + 6*beta)
M[0,1] = -ts*(alpha + 4*beta)
M[0,2] = ts*beta
M[0,-1] = -ts*(alpha + 4*beta)
M[0,-2] = ts*beta
N=M
 
for i in range(1,points):
    N = np.vstack((N,np.roll(M,i,axis=1)))
 
Q = np.linalg.inv(N)
 
#Q = np.matrix(Q)
 
steps = 9
h=[]
print out
vx = np.array(out[0])
vy = np.array(out[1])
 
bilx = interpimage.InterpImage(Fx)
bily = interpimage.InterpImage(Fy)
 
#plt.ion()
#fig = plt.figure()
 
#plt.imshow(F, cmap=plt.cm.Greys_r)
 
for step in range(steps):
    print "step ", step
    for i in range(vx.shape[0]):
        #print i,vx[i],vy[i]
        print bilx.bilinear(vy[i],vx[i]), bily.bilinear(vy[i],vx[i]), vx[i], vy[i]
        if bilx.bilinear(vy[i],vx[i]) < thrs:
            vx[i] = vx[i]-gamma*(bilx.bilinear(vy[i],vx[i]))
        if bily.bilinear(vy[i],vx[i]) < thrs:
            vy[i] = vy[i]-gamma*(bily.bilinear(vy[i],vx[i]))
 
        newstepx = np.dot(Q,vx)
        newstepy = np.dot(Q,vy)
  
 
def nextPlot(a):
    r.set_data(a[0],a[1])
    plt.draw()
    print "did some drawing"
    plt.show()
 
nextPlot(h[0])
time.sleep(1)
nextPlot(h[1])
time.sleep(1)
nextPlot(h[2])

