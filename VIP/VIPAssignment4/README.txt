To run this code you need these libraries installed

cv2
scipy
numpy
PIL
glob
pylab

Additionally, it requires the dudekface database as a subfolder.

To run our system, run the main.py which then imports lktrack.py. 

We implemented both OpenCV's tracker and our own.
OpenCV is in the function called CV_track_points()
Ours is in the function called our_track_points()

To switch between our tracker and OpenCV you have to change which function is called in 
the function called track() at the very end of ltkrack.py.

When OpenCV is chosen then line 27 in main.py must read: 
plt.plot([p[0] for p in t],[p[1] for p in t])

When our implementation is chosen then line 27 in main.py must read:
plt.plot([p[1] for p in t],[p[0] for p in t])


