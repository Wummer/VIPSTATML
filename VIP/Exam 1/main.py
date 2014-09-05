from __future__ import division
from PIL import Image
from scipy.ndimage import filters
from numpy import *
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean
from operator import itemgetter


#################################################################
#
#           Importing, converting and filtering images
#
#################################################################

plt.gray()
print "Importing pictures and converting them to arrays"
# Import images and transfer them into 2D float arrays
im1 = array(Image.open("imagedata/Img001_diffuse_smallgray.png"),dtype="float32")
im2 = array(Image.open("imagedata/Img002_diffuse_smallgray.png"),dtype="float32")
im3 = array(Image.open("imagedata/Img009_diffuse_smallgray.png"),dtype="float32")

imsq = Image.open("imagedata/squirrel.png").convert('L')
imot = Image.open("imagedata/otter.png").convert('L')
imsq = array(imsq,dtype="float32")
imot = array(imot,dtype="float32")

im1_gl = filters.gaussian_laplace(im1,sigma=1.4)
im2_gl = filters.gaussian_laplace(im2,sigma=1.4) # I have no real argument for sigma 1.4 besides it yields good results
im3_gl = filters.gaussian_laplace(im3,sigma=1.4)
imsq_gl = filters.gaussian_laplace(imsq,sigma=1.4)
imot_gl = filters.gaussian_laplace(imot,sigma=1.4)


print "Done."

"""detect(image)
This function finds the local extrema in a picture.
It runs through every pixel 
"""
def detect(image):
  extrema=[]

  for x in range(1,image.shape[0]-1):
      for y in range(1,image.shape[1]-1):
          if (image[x,y]< image[x,y+1]
          and image[x,y]< image[x,y-1]
          and image[x,y]< image[x-1,y]
          and image[x,y]< image[x+1,y]
          and image[x,y]<-30
          or               
          image[x,y]> image[x,y+1]
          and image[x,y]> image[x,y-1]
          and image[x,y]> image[x-1,y]
          and  image[x,y]> image[x+1,y]
          and image[x,y]>30):
              extrema.append([x,y])
              # We are only interested in finding the "extreme" salient objects as to diminish the amount of calculations.
  
  print "Extrema found: ",len(extrema)
  return extrema

##################################################################
#
#                   The functions
#
##################################################################


"""patch(image,width)
This function takes an image and a patch width and returns a patch for every extrema.
It calls  and returns  patch points in a list.
The last two values in the list is the interest point coordinates.

"""
def patch(image,width):
    wid = int(width / 2)
    patch=[]
    Patch=[]
    extrema=detect(image) #First we need the extrema!

    for point in extrema:

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




"""NCC(patch1,patch2)
This function takes two patches and returns the normalized cross correlation value for every match of patch1 and patch2. 
It utilizes numpy's mean and standard deviation functions (std, not to be confused with STDs!).

"""
def NCC(patch1,patch2):
    mean1,sta1= mean(patch1[:-2]), std(patch1[:-2]) # The last two are coordinates of the center
    mean2,sta2= mean(patch2[:-2]), std(patch2[:-2])
    sum=0

    for i in range(len(patch1)-2):
         sum+=((patch1[i]-mean1)*(patch2[i]-mean2))/(sta1*sta2) #The normalized cross correlation
    
    NCC=(sum/(len(patch1)-2))  
    return NCC



"""evaluate(one match point, list of possible match points)
Evaluate matches using scipy's euclidean distance, with one match point being matched against the entire list.
If the distance from the second best candidate from Match2 is more than 0.x of the best, the best is returned

"""
def evaluate(match1,Match2):  
    Distance = []

    for match2 in Match2:
        distance = euclidean(match2[:-2],match1[:-2]) #Omitting coordinates
        Distance.append([distance, match2])

    Distance = sorted(Distance, key=itemgetter(0)) #Sorted by the lowest distance first

    if Distance[0][0]/Distance[1][0]<0.1: #I presume the true matching pairs are really close to each other
        match = Distance[0][1] #Return only the best match
        return match    
    else: return []



"""match(patchlist1,patchlist2,threshold)
This function matches one patch from one image to all patches in the other image calling the NCC function.
If the squared NCC value is above the threshold T, patches from each list is stored in Matched1 and Matched2 at the same index and returned.

""" 
def match(patchlist1,patchlist2,threshold):
    Matched=[] 
    Matched1=[] 
    Matched2=[] 
    best_matched1=[]
    best_matched2=[]

    for patch1 in patchlist1:
        for patch2 in patchlist2:                   
            if (NCC(patch1,patch2))**2>threshold: #Effectively creating r**2 which means the NCC output is [0...1]
               Matched.append(patch2)
        if Matched!=[]:
            Matched2.append(Matched)
            Matched1.append(patch1)
        Matched=[]        

    for i in range(len(Matched1)):
        if len(Matched2[i])==1:
            best_matched1.append([Matched1[i][-2],Matched1[i][-1]]) #append coordinates           
            best_matched2.append([Matched2[i][0][-2],Matched2[i][0][-1]])
            
        elif len(Matched2[i])>1: # if more than one candidate: see if the best is convincing
            evaluation=evaluate(Matched1[i],Matched2[i])
            if evaluation!=[]: 
                best_matched1.append([Matched1[i][-2],Matched1[i][-1]]) #append coordinates
                best_matched2.append([evaluation[-2],evaluation[-1]])
                               
    return best_matched1,best_matched2


"""symanalysis(list of patches, list2 of patches, threshold)
This function does a two-way symmetrical analysis of the patches.
It only returns a match if point A and B have each other as their best match.

"""
def symanalysis(patchlist1,patchlist2,threshold):
    best1,best2=match(patchlist1,patchlist2,threshold)
    bestsym2, bestsym1 = match(patchlist2,patchlist1,threshold)

    real_best_matched1 =[] 
    real_best_matched2 =[] 
   
    for i in range(len(best1)):
        for sym1 in bestsym1:
            if best1[i] == sym1:
                real_best_matched1.append(best1[i])
                real_best_matched2.append(best2[i])

    return real_best_matched1, real_best_matched2


"""plot_matches(im1,im2,match1,match2)
This function plots the matched points from two different lists and draws a line from match1[0] to match2[0], match1[1] to match2[1] etc.

"""
def plot_matches(i1,i2,match1,match2):  
    i3 = concatenate((i1, i2), axis=1)
    plt.imshow(i3)
    cols = 800
    p = 0

    assert len(match1) == len(match2)
    for i in range (len(match1)):
        plt.plot([match1[i][1], match2[i][1]+cols], [match1[i][0], match2[i][0]], 'co-')
        p+=1
    
    plt.axis('off')
    plt.autoscale(False)
    plt.show()
    plt.close()




##########################################################
#
#       Detecting simple interest points
#
#######"##################################################

exsq = detect(imsq_gl)
exot = detect(imot_gl)

plt.subplot(221)
plt.imshow(imsq)
plt.axis('off')
plt.autoscale(True)
plt.title('Squirrel')
plt.subplot(222)
plt.imshow(imsq_gl)
plt.plot([p[1] for p in exsq], [p[0] for p in exsq], 'r.')
plt.axis('off')
plt.autoscale(True)
plt.title('Squirrel with interest points')


plt.subplot(223)
plt.imshow(imot)
plt.axis('off')
plt.autoscale(True)
plt.title('Otter')
plt.subplot(224)
plt.imshow(imot_gl)
plt.plot([p[1] for p in exot], [p[0] for p in exot], 'r.')
plt.axis('off')
plt.autoscale(True)
plt.title('Otter with interest points')

plt.show()
plt.close()


################################################################
#
#                       Image 1 & 2 calls
#
################################################################
print "----------------------------------------------"
print "Beginning feature extraction on image 1 and 2"
print "Finding interesting points and plotting them."

ex1 = detect(im1_gl)
ex2 = detect(im2_gl)

plt.subplot(121)
plt.imshow(im1)
plt.plot([p[1] for p in ex1], [p[0] for p in ex1], 'r.')
plt.axis('off')
plt.autoscale(False)

plt.subplot(122)
plt.imshow(im2)
plt.plot([p[1] for p in ex2], [p[0] for p in ex2], 'r.')
plt.axis('off')
plt.autoscale(False)

plt.show()
plt.close()

print "Done."

print "Calculating patches."
patchlist1=patch(im1_gl,15) 
patchlist2=patch(im2_gl,15)
print "Done."

print "Finding matches between image 1 and 2 with symmetrical analysis."
Points1,Points2=symanalysis(patchlist1,patchlist2,threshold=0.8) #Again we see the result of the engineering approach. No arguments, all results!
print "Done."

print "Plotting matches."
plot_matches(im1,im2,Points1,Points2)    
    

###############################################################
#
#                       Image 1 & 3
#
###############################################################

print "----------------------------------------------"
print "Beginning feature extraction on image 1 and 3"
print "Calculating patches."
patchlist2=patch(im3_gl,15)
print "Done."

print "Finding matches between image 1 and 3 with symmetrical analysis."
Points3,Points4=symanalysis(patchlist1,patchlist2,threshold=0.8) #Again we see the result of the engineering approach. No arguments, all results!
print "Done."

print "Plotting matches."
plot_matches(im1,im3,Points3,Points4)        