from __future__ import division
from PIL import Image
from scipy.ndimage import filters
from numpy import *
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean


plt.gray()
# Import images and tranfer them into 2D arrays
#image1 = array(Image.open("imagedata/Img001_diffuse_smallgray.png"))
#image2 = array(Image.open("imagedata/Img002_diffuse_smallgray.png"))
#image3 = array(Image.open("imagedata/Img009_diffuse_smallgray.png"))

image1 = (Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img001_diffuse_smallgray.png"))
image2 = (Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img002_diffuse_smallgray.png"))
image9 = (Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img009_diffuse_smallgray.png"))


# Filter image with Gussian Filter
image_GF= filters.gaussian_laplace(image1, sigma=3)#Laplacian Gussian filter
image_GF2= filters.gaussian_laplace(image2, sigma=3)
image_GF9= filters.gaussian_laplace(image9, sigma=3)


def detect(image):
  
  extremas=[]
  for x in range(1,image.shape[0]-1):
      for y in range(1,image.shape[1]-1):
          if (image[x,y]< image[x,y+1]
          and image[x,y]< image[x,y-1]
          and image[x,y]< image[x-1,y]
          and image[x,y]< image[x+1,y]
          and image[x,y]<5
          or               
          image[x,y]> image[x,y+1]
          and image[x,y]> image[x,y-1]
          and image[x,y]> image[x-1,y]
          and  image[x,y]> image[x+1,y]
          and image[x,y]>250):
              extremas.append([x,y])
  print len(extremas)
  return extremas

def patch(image,width):
#This function takes an image and a patch width and returns 
#the gray value of all patch points in a list. The last two 
# values in the list is the interest point coordinates
    wid = int(width / 2)
    patch=[]
    Patch=[]
    extremas=detect(image)
    for point in extremas:
       for i in range(point[0]-wid,point[0]+wid+1):
           for j in range(point[1]-wid,point[1]+wid+1):
               patch.append(image[i,j])
       
       patch.append(point[0])
       patch.append(point[1])
       Patch.append(patch)
       patch=[]
    
    return Patch
"""    
def compute(Patch):#here patch looks like [value, value, value... x, y]
    sum=0
    sum2=0
    for i in range(len(Patch)-2):
            sum+=Patch[i]
    meanvalue=sum/(len(Patch)-2)
    
    for j in range(len(Patch)-2):
        sum2+=(Patch[j]-meanvalue)**2    
    standard=sqrt(sum2/(len(Patch)-2))
    
    return meanvalue,standard
"""
def NCC(patch1,patch2):
#A function that takes two patches and returns the
#normalized cross correlation value for every match
    mean1,sta1= mean(patch1[:-2]), std(patch1[:-2]) # The last two are coordinates of the center
    mean2,sta2= mean(patch2[:-2]), std(patch2[:-2])

    #mean1,sta1= compute(patch1)
    #mean2,sta2= compute(patch2)    
    sum=0
    for i in range(len(patch1)-2):
         sum+=((patch1[i]-mean1)*(patch2[i]-mean2))/(sta1*sta2)        
    NCC=(sum/(len(patch1)-2))  
    return NCC
 
def evaluate(Match1,Match2):
#Euclidean distance. Match 1 is one patch. Match 2 is a list of possible matches. 
#if the distance from the best candidate from Match2 is more than 0.8 of than the second best
#evaluate using bubble sort
    sum=0
    Distance=[]
    for match2 in Match2:
        for i in range(len(Match1)-2):
           sum+=(Match1[i]-match2[i])**2
        distance=sqrt(sum)
        Distance.append(distance)
        sum=0
    
    for key in range(len(Match2)-1):
       for index in range(len(Match2)-key-1):
          if  Distance[index]>Distance[index+1]: 
               Distance[index],Distance[index+1]=Distance[index+1],Distance[index]
               Match2[index],Match2[index+1]=Match2[index+1],Match2[index]
    
    if Distance[0]/Distance[1]<=0.5:
        return Match2[0]
    else: return []
"""
def evaluate(Match1,Match2):
#Evaluate matches using euclidean distance. Match 1 is one patch. Match 2 is a list of possible matches. 
#if the distance from the best candidate from Match2 is more than 0.8 of 
    sum=0
    Distance=[]
    for match2 in Match2:
        distance = euclidean(Match1[:-2], match2[:-2])
        Distance.append(distance)
        sum = 0        
    Distance = sorted(Distance)
    if Distance[0]/Distance[1]<=0.8:
        return Match2[0]
    else: return []

 """
def match(patchlist1,patchlist2,threshold):# Question: Is it possible two points in image1 match the same point in image2
    #This function matches one patch from one image to all patches in the other image
    # calling the NCC function. If the ncc value is above the threshold
    #patches from each list is stored in Matched1 and Matched2 with the same index number
    #If the best match from Match2 is more than T closer than the second best, this function returns the two matches in 
    #the lists best_matched1 and best_matched2
    Matched=[] #temporary list
    Matched1=[] # a 1D list of patches from patchlist1 that has matches above the threshold from list 2
    Matched2=[] # a list of list of (more) patches from patchlist 2 that matches one patch list1
    best_matched1=[] # a list of best matches from image1
    best_matched2=[] # a list of best matches from image2
    for patch1 in patchlist1:
        for patch2 in patchlist2:                   
            if NCC(patch1,patch2)>threshold:
               Matched.append(patch2)
        if Matched!=[]:
            Matched2.append(Matched)
            Matched1.append(patch1)
        Matched=[]        
    # evaluate if the best match is convincing
    for i in range(len(Matched1)):
        if len(Matched2[i])==1: # if there is only one candidate
            best_matched1.append([Matched1[i][-2],Matched1[i][-1]]) #append coordinates           
            best_matched2.append([Matched2[i][0][-2],Matched2[i][0][-1]]) # append coordinates
            
        elif len(Matched2[i])>1: # if more than one candidate: see if the best in convincing
            evaluation=evaluate(Matched1[i],Matched2[i])
            if evaluation!=[]: 
                best_matched1.append([Matched1[i][-2],Matched1[i][-1]])
                best_matched2.append([evaluation[-2],evaluation[-1]])
                               
    return best_matched1,best_matched2

patchlist1=patch(image_GF, 3)
patchlist2=patch(image_GF2, 3)

Points1,Points2=match(patchlist1,patchlist2,threshold=0.5)

def plot_matches(image1,image2,match1,match2): 
# This function plots the matched points from two different lists and draws a
#line from match1[0] to match2[0], match1[1] to match2[1] etc.  
    image3 = concatenate((image1, image2), axis=1)
    plt.imshow(image3)
    cols = 800
    #cols = image1.shape[1]#800
    p = 0
    assert len(match1) == len(match2)
    for i in range (len(match1)):
        plt.plot([match1[p][1], match2[p][1]+cols], [match1[p][0], match2[p][0]], 'co-')
        p+=1
    plt.axis('off')
    plt.autoscale(False)
    plt.show()

plot_matches(image1,image2,Points1,Points2)    
    
plt.subplot(1,2,1)
plt.imshow(image1)
plt.plot([p[1] for p in Points1], [p[0] for p in Points1], 'r.')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image2)
plt.plot([p[1] for p in Points2], [p[0] for p in Points2], 'r.')
plt.axis('off')
plt.show()



