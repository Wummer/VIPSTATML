from __future__ import division
from PIL import Image
from scipy.ndimage import filters
from numpy import *
import matplotlib.pyplot as plt
import math

plt.gray()
# Import images and tranfer them into 2D arrays
image1 = (Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img001_diffuse_smallgray.png"))
image2 = (Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img002_diffuse_smallgray.png"))
#image1 = array(Image.open("imagedata/Img001_diffuse_smallgray.png"))
# image2 = array(Image.open("imagedata/Img002_diffuse_smallgray.png"))


# Filter image with Gussian Filter
image_GF= filters.gaussian_laplace(image1, sigma=5)#Laplacian Gussian filter
image_GF2= filters.gaussian_laplace(image2, sigma=5)
neighbor = 4#the number of neighbor is 4

def detectinterest(image_GF):
  localmin=[]
  localmax=[]
  interest=[]
  for x in range(1,image_GF.shape[0]-1):
    for y in range(1,image_GF.shape[1]-1):
         if image_GF[x,y]< image_GF[x,y+1] and image_GF[x,y]< image_GF[x,y-1] and image_GF[x,y]< image_GF[x-1,y] and  image_GF[x,y]< image_GF[x+1,y] and image_GF[x,y]<5:
                localmin.append([x,y])
                interest.append([x,y])                
         if image_GF[x,y]> image_GF[x,y+1] and image_GF[x,y]> image_GF[x,y-1] and image_GF[x,y]> image_GF[x-1,y] and  image_GF[x,y]> image_GF[x+1,y] and image_GF[x,y]>250:
                localmax.append([x,y])
                interest.append([x,y])
  return interest



def patch(image):
# does it need to be fixed to omit patches too close to edges
    patch=[]
    Patch=[]
    interest=detectinterest(image)
    for point in interest:
       for i in range(point[0]-1,point[0]+2):
           for j in range(point[1]-1,point[1]+2):
               patch.append(image[i,j])
       
       patch.append(point[0])
       patch.append(point[1])
       Patch.append(patch)
       patch=[]
    
    return Patch
    
def compute(Patch,image):#here patch looks like [value, value, value... x, y]
    sum=0
    sum2=0
    for i in range(len(Patch)-2): 
            sum+=Patch[i]
    meanvalue=sum/(len(Patch)-2)
    
    for j in range(len(Patch)-2):
        sum2+=(Patch[j]-meanvalue)**2    
    standard=sum2/(len(Patch)-2)
    
    return meanvalue,standard

def NCC(patch1,patch2):
    mean1,sta1=compute(patch1,image_GF)
    mean2,sta2=compute(patch2,image_GF2)
    sum=0
    for i in range(len(patch1)-2):
         sum+=((patch1[i]-mean1)*(patch2[i]-mean2))/(sta1*sta2)
         
    NCC=1-(sum/(len(patch1)-2))     # check Discussion board
    return NCC
    
def evaluate(Match1,Match2):#here match1 is patch and match2 is something like [patch2, patch2, patch2]
    sum=0
    Distance=[]
    for match2 in Match2:
        for i in range(len(Match1)-2):
           sum+=(Match1[i]-match2[i])**2 # here you square them
        distance=sqrt(sum) #and here you take the squareroot
        Distance.append(distance)
        sum=0
    
    for key in range(len(Match2)-1):
       for index in range(len(Match2)-key-1):
          if  Distance[index]>Distance[index+1]:
               Distance[index],Distance[index+1]=Distance[index+1],Distance[index]
               Match2[index],Match2[index+1]=Match2[index+1],Match2[index]
    
    if Distance[0]/Distance[1]<=0.8:
        return Match2[0]
    else: return []
    
def match(patchlist1,patchlist2,threshold):# Question: Is it possible two points in image1 match the same point in image2
    Matched=[]
    Matched1=[]
    Matched2=[]
    for patch1 in patchlist1:
        for patch2 in patchlist2:                   
            if NCC(patch1,patch2)>threshold:
               Matched.append(patch2)
        if Matched!=[]:
            Matched2.append(Matched)
            Matched1.append(patch1)
        Matched=[]
        
    # evaluate if the mathed interests are good matched interests        
    real_matched1=[]
    real_matched2=[]
    for index in range(len(Matched1)):
        if len(Matched2[index])==1:
            real_matched1.append([Matched1[index][9],Matched1[index][10]])            
            real_matched2.append([Matched2[index][0][9],Matched2[index][0][10]])
            
        elif len(Matched2[index])>1:
                 evaluation=evaluate(Matched1[index],Matched2[index])
                 if evaluation!=[]: 
                     real_matched1.append([Matched1[index][9],Matched1[index][10]])
                     real_matched2.append([evaluation[9],evaluation[10]])
            
                     
    return real_matched1,real_matched2

patchlist1=patch(image_GF)
patchlist2=patch(image_GF2)

Points1,Points2=match(patchlist1,patchlist2,threshold=0.5)

ncc= NCC(patchlist1, patchlist2)
print ncc

def plot_matches(image1,image2,match1,match2): 
# This function plots the matched points from two different lists and draws a
#line from match1[0] to match2[0], match1[1] to match2[1] etc.  
    image3 = concatenate((image1, image2), axis=1)
    plt.imshow(image3)
    cols = 800
    #cols = im1.shape[1]#800
    p = 0
    assert len(match1) == len(match2)
    for i in range (len(match1)):
        plt.plot([match1[p][1], match2[p][1]+cols], [match1[p][0], match2[p][0]], 'co-')
        p+=1
    plt.axis('off')
    plt.autoscale(False)
    plt.show()
# Draw matched points not all interest points could be matched well

plot_matches(image1,image2,Points1,Points2)

print Points2
    
plt.subplot(1,2,1)
plt.imshow(image1, origin='lower')
plt.plot([p[1] for p in Points1], [p[0] for p in Points1], 'r.')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image2, origin='lower')
plt.plot([p[1] for p in Points2], [p[0] for p in Points2], 'c.')
plt.axis('off')
plt.show()

