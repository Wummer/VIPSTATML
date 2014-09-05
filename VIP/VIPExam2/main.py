from __future__ import division
from math import *
from bs4 import BeautifulSoup
from collections import Counter
from scipy.cluster.vq import kmeans,vq,whiten
from PIL import Image
from operator import itemgetter
import cv2
import numpy as np
import glob
import pylab as pl
import ast
import pickle

# Extracting test and train set

print "=" * 60
print "Initializing the script.\n"

path1 = glob.glob('../*/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('../*/101_ObjectCategories/brontosaurus/*.jpg')

train1 = path1[:30]
train2 = path2[:30]
train1.extend(train2)

test1 = path1
test2 = path2
test1.extend(test2)

# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()
k = 3000

#--------------------------------------------------------------------------
#Detection and bag of visual words

"""detectcompute(data)
This function takes a list of image paths as input.
It then calculates the SIFT descriptors for the entire data input and returns all descriptors as rows in a single array.
"""
def detectcompute(data):
	descr = []

	print "="*60,"\n"
	print "Locating all SIFT descriptors for the train set."

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		descr.append(des)
	
	out = np.vstack(descr) #Vertical stacking of our descriptor list. Genius function right here.
	print "Done.\n"
	return out



"""singledetect(data)
This function takes a list of image paths as inputs.
It then outputs each images' path and corresponding SIFT descriptors.
"""
def singledetect(data):
	sd = []
	print "Locating and assigning SIFT descriptors for each image"

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		sd.append([data[i],des])

	print "Done.\n"
	return sd



"""bow(list of images,codebook,clusters)
This function taskes a list of image paths, a codebook and an integer denoting the amount of clusters as input.
It then computes each image's bag of words as a normalized histogram in a pseudo-dictionary.
It then outputs the image path, their descriptors and their normalized histogram.
"""
def bow(images,codebook,clusters):
	out = images
	temp = []

	print "-"*60
	print "Creating the pseudo database."
	for im in images:
		c = Counter()
		bag,dist = vq(whiten(im[1]),codebook)
		
		for word in bag:
			c[word]+=1

		#Creating histograms
		for i in range(clusters):
			if i in c.iterkeys():
				c[i] = c[i]/sum(c.values())
			if i not in c.iterkeys():
				c[i] = 0
		
		temp.append(c)
		
	for i in range(len(temp)):
		out[i].append(temp[i])

	print "Done.\n"
	return out



#--------------------------------------------------------------------------
#Creating database and writing to files

"""createdatabase()
This function takes no direct input, but utilizes the image paths assigned at the beginning of the script.
It then computes the K-means clustering on the specified training set and calculates vector quantization on every images descriptor up against the clusters.
It then calls the Bag-of-Visual-Words function. Finally it outputs every image's path and normalized histogram into one file, and the codebook into another file.
"""
def createdatabase():
	X_train = detectcompute(train1)

	print "Clustering the data with K-means"
	codebook,distortion = kmeans(whiten(X_train),k)
	print "Done.\n"
	
	imtrain = singledetect(test1)
	Pdatabase = bow(imtrain,codebook,k) #Pseudo database with list structure


	#Writing to html.table
	print "Converting the database into a HTML file"
	htmltable = open("table.htm","r+") 
	begin = "<htm><body><table cellpadding=5><tr><th>Filename</th><th>Histogram</th></tr>"
	htmltable.write(begin)

	for i in range(len(Pdatabase)):
	    middle = "<tr><td>%(filename)s</td><td>%(histogram)s</td></tr>" % {"filename": Pdatabase[i][0], "histogram": Pdatabase[i][-1]}
	    htmltable.write(middle)

	end = "</table></body></html>"    
	htmltable.write(end)
	htmltable.close()
	print "Done.\n"

	codebook_to_file(codebook)



"""
codebook_to_file()
This function saves the codebook to a file.
"""
def codebook_to_file(codebook):
	print "Saving codebook to file"
	codebookfile = open("codebook.txt", "r+")
	pickle.dump(codebook, codebookfile)
	
	codebookfile.close()
	print "Done.\n"



"""
codebook_from_file()
This function retrieves the codebook from the file.
It returns the codebook. 
"""

def codebook_from_file():
	from_db = open("codebook.txt", "r")
	codebook_from_db = pickle.load(from_db)

	from_db.close()
	return codebook_from_db



"""
from_database(path_to_db, filename)
This function retrieves and returns everything from our database: filenames and the adjacent histogram.
These are structured in a nested list like this: [[filename, hisogram],[filename,histogram]...]
"""
def from_database():
	database =[]
	htmldoc = open("table.htm","r") 
	db = BeautifulSoup(htmldoc)
	table = db.find('table')
	rows = db.findAll('tr')
	
	for row in rows[1:]:
		temp = []
		filename = row.find('td') 
		temp.append(filename.text)
		hist = filename.findNext('td') 
		temp.append(ast.literal_eval(hist.text[7:]))
		database.append(temp)

	htmldoc.close()
	return database


#--------------------------------------------------------------------------
#Retrieval measures

"""userretrieval()
This function prompts the user for an input, which it then tries to convert into a string.
This function calls Bhattacharyya and commonwords on the user specified image.
It then calls the output of those functions with present_results.
"""
def userretrieval():
	print "\nTo create a query, please select a value ranging from 0-83\n"
	print "Lobster images are located between 0-41"
	print "Brontosaurus images are located between 42-83"
	print "-"*45,"\n"

	passed = False
	while passed is False:
		try: 
			uquery = int(raw_input("Choose an image: "))
			passed = True
		except ValueError:
			print "Invalid input. Not a number."

	inside = False
	if uquery < 0 or uquery > 83:
		print "Invalid input. Please try again"
		while inside is False:
			try:
				query = int(raw_input("Choose an image: "))
				inside = True
			except ValueError:
				print "Invalid input. Not a number"

	db = from_database()
	im = db[uquery]
	print len(db)
	bhat = Bhattacharyya(im,db)
	present_results(im,bhat,"Bhattacharyya")

	freq = commonwords(im,db)
	present_results(im,freq,"Common Words")


"""Bhattacharyya(one query image, a database)
This function takes a single image and an image database as its input.
It then tries to match the image with every image in the database by measuring the Bhattacharyyan distance between them.
It then returns the 9 closests matches.
"""
def Bhattacharyya(queryimage,db):
    count=[]

    amount=0
    for num in range(len(db)):
        for i in range(k):
           amount+=sqrt(queryimage[1][i]*db[num][1][i])
        count.append([amount,db[num][0]])
        amount=0

    Result=sorted(count,key=itemgetter(0),reverse = True)
                     
    queryresult=[]
    for j in range(len(Result)):
        queryresult.append(Result[j][1])#input the path to queryrsult

    return queryresult



"""commonwords(a query image, database)
This function takes an input of a single image and the database.
It then compares the content of the query image to all the images in the database. 
For every class/word that is in both inputs, it will add 1.
It returns a list of integers ranging from 0-k, where k is most alike, with a lenght of the database.
"""
def commonwords(queryimage,db):
    Common=[]
    queryresult=[]
    for num in range(len(db)):
        count=0
        for i in range(k):
           if queryimage[1][i]!= 0 and db[num][1][i]!= 0:
                count+=1
        Common.append([count,db[num][0]])
    Result=sorted(Common,key=itemgetter(0),reverse=True)#Descending
                      
    for j in range(len(Result)):
        queryresult.append(Result[j][1])

    return queryresult



"""
 present_results(queryimage, resultpath, similarityfunction)
 This function takes the queryimage, the list of filenames for the 9 best matched images and the name of the similarity function.
 It prints the precision rate and plots the queryimage and the matched images
"""   
def present_results(queryimage, resultpath, similarityfunction):
	counter = 0
	l = queryimage[0].find("lobster")
	b = queryimage[0].find("brontosaurus")

	if l < 0: 
		label = "brontosaurus"
	else:
		label = "lobster"

	for r in resultpath[:9]:
		if r.find(label) > 0:
			counter +=1
			result = counter / len(resultpath[:9])

	print ('Result for %s \n' %queryimage[0])
	print ('%s:' % similarityfunction)
	print ('Precision rate in top 9: %s' %result)

	imageplot=[]

	for result in resultpath:
	    img=np.array(Image.open(result))
	    imageplot.append(img)

	#plot the query image
	pl.imshow(np.array(Image.open(queryimage[0])))
	pl.axis('off')
	pl.show()

	#plot the matched images
	for i in range(9):
	    pl.subplot(331+i)
	    show = i+1
	    pl.imshow(imageplot[i])
	    pl.title('%s' % show)
	    pl.axis('off')

	pl.suptitle('9 best of %(sim)s on %(label)s' % {'sim':similarityfunction,'label': label})

	pl.show()
	pl.close()



#--------------------------------------------------------------------------
#Interface

"""userinput()
This function is called at the beginning and takes a user input.
The input is then used as an ouput to call the commands(cmd) function.
"""
def userinput():
	print "="*60
	print "Please select one of the following 3 options. \n"
	print "-"*45
	print "1. Compute the database."
	print "2. Retrieve an image."
	print "3. Exit"
	print "-"*45
	usercmd = raw_input("Choose an option: ")
	commands(usercmd)



"""commands(cmd)
This function takes an integer as input.
Depending on what integer is given, the function will either call createdatabase, userretrieval or SystemExit().
When that function is done, it will call userinput() again.
"""
def commands(cmd):
	legal = ["1","2","3"]

	if cmd not in legal:
		print "Invalid input. Please use the numerical value.\n"
		userinput()

	elif cmd == "1":
		createdatabase()
		print "A database has been created. \n" 
		userinput()

	elif cmd == "2":
		userretrieval()
		print "A retrieval has been made."
		userinput()

	elif cmd == "3":
		print "Quit succesfully."
		raise SystemExit()



"""main()
Starts the programme by calling the userinput-function
"""
def main():
    print ">>> Content Based Image Retrieval by Maria, Guangliang and Alexander \n Vision and Image Processing assignment 2 \n";
    userinput();


if __name__ =='__main__':
    main(); 
