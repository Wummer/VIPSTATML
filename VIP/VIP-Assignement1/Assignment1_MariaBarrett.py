from __future__ import division
from PIL import Image
from scipy.ndimage import filters
from numpy import *
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean
from operator import itemgetter

plt.gray()

# Import images and tranfer them into 2D arrays of floats

myimage1 = array(Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/human.png"),dtype='float32')
myimage2 = array(Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/building.png"),dtype='float32')


image1 = array(Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img001_diffuse_smallgray.png"),dtype='float32')
image2 = array(Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img002_diffuse_smallgray.png"),dtype='float32')
image9 = array(Image.open("/Users/Maria/Documents/ITandcognition/bin/Images/Img009_diffuse_smallgray.png"),dtype='float32')

#image1 = array(Image.open("Images/Img001_diffuse_smallgray.png"))
#image2 = array(Image.open("Images/Img002_diffuse_smallgray.png"))
#image9 = array(Image.open("Images/Img009_diffuse_smallgray.png"))
#myimage1 = array(Image.open("Images/human.png"),dtype='float32')
#myimage2 = array(Image.open("Images/building.png"),dtype='float32')


# Filter image with Gussian Filter
image_GF= filters.gaussian_laplace(image1, sigma=1.4)#Laplacian Gussian filter
image_GF2= filters.gaussian_laplace(image2, sigma=1.4)
image_GF9= filters.gaussian_laplace(image9, sigma=1.4)

myimage1_GF= filters.gaussian_laplace(myimage1, sigma=1.4)#Laplacian Gussian filter
myimage2_GF= filters.gaussian_laplace(myimage2, sigma=1.4)


def detect(image, threshold):
  extremas=[]
  for x in range(1,image.shape[0]-1):
      for y in range(1,image.shape[1]-1):
          if image[x,y]< image[x,y+1] and image[x,y]< image[x,y-1] and image[x,y]< image[x-1,y] and image[x,y]< image[x+1,y] and image[x,y]< -threshold: #local minima
              extremas.append([x,y])                
          if image[x,y]> image[x,y+1] and image[x,y]> image[x,y-1] and image[x,y]> image[x-1,y] and  image[x,y]> image[x+1,y] and image[x,y]>threshold: #local maxima
              extremas.append([x,y])
  print ("number of extremas %s" % len(extremas))
  return extremas


def patch(image,width):
#This function takes an image and a patch width and returns 
#the gray value of all patch points in a list. The last two 
# values in the list are the interest point coordinates
    wid = int(width / 2)
    patch=[]
    Patch=[]
    extremas=detect(image, threshold=20)
    for point in extremas:
        if point[0]-wid > 0 and point[0]+wid < len(image): #omit patch if if it cannot fit within image borders
            if point[1]-wid > 0 and point[1]+wid < len(image[0]):
               for i in range(point[0]-wid,point[0]+wid+1):
                   for j in range(point[1]-wid,point[1]+wid+1):
                       patch.append(image[i,j])
               
               patch.append(point[0])
               patch.append(point[1])
               Patch.append(patch)
               patch=[]
            
    return Patch

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
    ncc=(sum/(len(patch1)-2))  
    return ncc


def evaluate(match1,Match2):
#Evaluate matches using euclidean distance. match1 is one patch. Match2 is a list of possible matches. 
#if the distance from the second best candidate from Match2 is more than 0.x of the best, the best
#is returned  
    Distance = []
    for match2 in Match2:
        distance = euclidean(match2[:-2],match1[:-2]) #not including the last two coordinate values
        Distance.append([distance, match2])
    Distance = sorted(Distance, key=itemgetter(0)) 
    if Distance[0][0]/Distance[1][0]<=0.5: #only append if distance of best is a lot better than second best
        ifoundamatch = Distance[0][1] #equal to match2
        return ifoundamatch    
    else: return []

def match(patchlist1,patchlist2,threshold):
    #This function matches one patch from one image to all patches in the other image
    # calling the NCC function. If the ncc value is above the threshold
    #patches from each list is stored in Matched1 and Matched2 with the same index number
    #If the best match from Match2 is more than 0.x closer than the second best, then  function returns the two matches in 
    #the lists best_matched1 and best_matched2
    Matched=[] #temporary list
    matched1=[] # a 1D list of patches from patchlist1 
    Matched2=[] # a list of list of (more) patches from patchlist 2 that matches one patch from patchlist 1
    best_matched1=[] # a list of best matches from image1
    best_matched2=[] # a list of best matches from image2
    
    for patch1 in patchlist1:
        for patch2 in patchlist2:                   
            if (NCC(patch1,patch2))**2>threshold:
               Matched.append(patch2)
        if Matched!=[]:
            Matched2.append(Matched)
            matched1.append(patch1)
        Matched=[]

    # evaluate if the best match is convincing
    for i in range(len(matched1)):
        if len(Matched2[i])==1: # if there is only one candidate
            best_matched1.append([matched1[i][-2],matched1[i][-1]]) #append coordinates           
            best_matched2.append([Matched2[i][0][-2],Matched2[i][0][-1]]) # append coordinates
            
        elif len(Matched2[i])>1: # if more than one candidate: see if the best in convincing
            evaluation=evaluate(matched1[i],Matched2[i])
            if evaluation!=[]: 
                best_matched1.append([matched1[i][-2],matched1[i][-1]])
                best_matched2.append([evaluation[-2],evaluation[-1]])
    print ("best matches one sided: %s" % len(best_matched1))
    return best_matched1, best_matched2


def symmetri(patchlist1, patchlist2):
#This function calls the big match function twice. It ensures 1) that the outputted coordinates are teh best match whether you start with image 1 or image2
# and 2) that an interest point in one image cannot be matched to more points in the other image. 

    best1,best2=match(patchlist1,patchlist2,threshold=0.5)
    bestsym2, bestsym1 = match(patchlist2,patchlist1,threshold=0.5)
    
    real_best_matched1 =[] #a list of symmetrical best matches. Only append to this list if the best match is found symetrically too
    real_best_matched2 =[] #a list of symmetrical best matches. Only append to this list if the best match is found symetrically too
   
    #check wheter a match occurs in the symmetrical list too
    for i in range(len(best1)):
        for sym1 in bestsym1:
            if best1[i] == sym1:
                real_best_matched1.append(best1[i])
                real_best_matched2.append(best2[i])

    return real_best_matched1, real_best_matched2

#-----------------------------------------------------------------------------------------------
#Calling
"""
patchsize=[2,4,6,8]
for p in patchsize:
    patchlist1=patch(image_GF, 4)
    patchlist2=patch(image_GF2, 4)


"""
patchlist1=patch(image_GF, 8)
patchlist2=patch(image_GF2, 8)


Points1,Points2 = symmetri(patchlist1, patchlist2)
print ("symmetrical best matches %s" % len(Points1))

#-----------------------------------------------------------------------------------------------
#Plotting

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
"""    
plt.subplot(1,2,1)
plt.plot([p[1] for p in Points1], [p[0] for p in Points1], 'r.')
plt.imshow(image1)
#plt.gca().invert_yaxis()
plt.axis('off')

plt.subplot(1,2,2)
plt.plot([p[1] for p in Points2], [p[0] for p in Points2], 'r.')
plt.imshow(image2)
#plt.gca().invert_yaxis()
plt.axis('off')
plt.show()

#----------------------------------------------------------------
#Plotting myimages with extremas

extremas1 = detect(myimage1_GF, 11)
extremas2 = detect(myimage2_GF, 24)

plt.subplot(221)
plt.imshow(myimage1)
plt.axis('off')
plt.autoscale(False)
plt.title('Humans')



plt.subplot(223)
plt.imshow(myimage1_GF)
plt.plot([p[1] for p in extremas1], [p[0] for p in extremas1], 'c.')
plt.axis('off')
plt.autoscale(False)
plt.title('Humans LoG with extremas, threshold 11')


plt.subplot(222)
plt.imshow(myimage2)
plt.axis('off')
plt.autoscale(False)
plt.title('Building')


plt.subplot(224)
plt.imshow(myimage2_GF)
plt.plot([p[1] for p in extremas2], [p[0] for p in extremas2], 'c.')
plt.axis('off')
plt.autoscale(False)
plt.title('Building LoG with extremas, threshold 24')
plt.show()

"""
