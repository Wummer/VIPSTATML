from __future__ import division
import lktrack 
import glob
import pylab as plt
from PIL import Image


imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames,reverse=True)

print "Calculating."
#create tracker object
lkt = lktrack.LKTracker(imnames[:5])

ims = []
for im,ft in lkt.track():
	print 'tracking %d features' % len(ft)

# plot the tracks
plt.imshow(im) 
for p in ft:
	plt.plot(p[1],p[0],'bo') #Use this for our
	#plt.plot(p[0],p[1],'bo') #Use this for OpenCV
for t in lkt.tracks:
	plt.plot([p[1] for p in t],[p[0] for p in t],'r-') #use this for our implementation
	#plt.plot([p[0] for p in t],[p[1] for p in t]) #use this for OpenCV
plt.axis('off')
plt.show()