#!/usr/bin/python
# -*- coding: utf-8 -*-


############################################################################################################################################
#
#	PyStab. (Jan. 2016)
#	=============================== 
# 	After an original idea of L. Dupuy and insitu7 c++ script (laurent.dupuy@cea.fr)
#	Requirements: python 2.7, and OpenCV 3.0.0
#
# 	Use:
# 	python pystab.py -i mymovie.avi :to navigate in the video and draw roi. Use the trackbar + any key to navigate in the movie; press, 
# drag and release to draw a rectangle + any key to remove; ESC to exit. The x,y,w,h (position and size) of the roi are printed (can be copied
# and paste in the xml file)
# 
#	python pystab.py -s myfile.xml : to stabilize the video in a specific area (roi) given a fixed area (template). Output: a series of
#jpg pictures of the stabilized movie, a video preview (opt). If ffmpeg enable, an output movie can be generated.
# 
# myfile.xml describes: 
# Mandatory: roi and template x,y,w,h (position and size)
# Optional: 
# - newmovie: starting and ending points of the output movie. Step (especially for long video)
# - spot: remove camera imperfection either by removing the background image obtained during a rapid camera motion (set by start and end) or by inpainting
# the spot by defining the spot location(x,y,w,h) and the inpainting radius (inp). Enable if inpainting is set to "yes"
# - enhance:  improve the video image. denoise: time averaging to remove noise. smooth: smooth out the image trajectory given a certain tolerance (deviate)
# (trajectory.png shows the x and y motion and the corresponding smoothed trajectory)
# - contrast: Improve contrast by making Contrast Limited Adaptive Histogram Equalization
# - scalebar for adding a scale (mag: image magnification x1000, size in nm)
# - ffmpeg: enable the ffmpeg to export a movie. deinterlace option enable if "yes". Output specifies the video name.
# - preview: enable video preview of the raw, stabilized and enhanced video if the parameters are set to "yes"
################################################################################################################################################""



from __future__ import division
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import argparse
from matplotlib import pyplot as plt
import subprocess
import timeit
from datetime import datetime

a = datetime.now()


#####################
#
# Parse arguments
#
#####################


parser = argparse.ArgumentParser()

parser.add_argument("-s", action="store_true", help="Stabilize the video, remove camera spot and enhance image. Use the values for roi and template using the -i option")
parser.add_argument("-i", action="store_true", help="Display the video frame, draw rectangles and get roi position and size. Use the trackbar + any key to navigate in the movie; press, drag and release to draw a rectangle + any key to remove; ESC to exit. The x,y,w,h (position and size) of the roi are printed")
parser.add_argument("filename", help=" with -i option: a movie file. with -s option a xml file with: [MANDATORY] a roi and a template x,y,w,h (position and size). [OPTIONAL]: a starting and ending point of the video, a spot argument for camera imperfection removal, enhance for denoising and trajectory smoothing, contrast for Contrast Limited Adaptive Histogram Equalization, scalebar, and ffmpeg for exporting video, and preview. Output: a series of pictures")
args = parser.parse_args()


#####################
#
# -i option
#
#####################


if args.i:
	vidcap = cv2.VideoCapture(args.filename)
	ret,frame = vidcap.read() #read the video
	def nothing(x):
		pass
	cv2.namedWindow(str(args.i))
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) #get movie number of frame
	fps= vidcap.get(cv2.CAP_PROP_FPS) #get fps
	cv2.createTrackbar('frame',str(args.i),0,length,nothing) #create a trackbar to navigate
	refPt = []
	
	##################################################################################################
	# roi draw function from  
	# http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
	#################################################################################################
	
	 
	def roi_draw(event, x, y, flags, param):
		# grab references to the global variables
		global refPt
	 
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			refPt = [(x, y)]
			
	 
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			refPt.append((x, y))
			
	 
			# draw a rectangle around the region of interest
			cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 1)
			#Print position x,y and size w,h
			print('x="'+str(refPt[0][0])+'"'+' y="'+str(refPt[0][1]))+'"'+' w="'+str(np.abs(refPt[1][0]-refPt[0][0]))+'"'+' h="'+str(np.abs(refPt[1][1]-refPt[0][1]))+'"'
			cv2.imshow(str(args.i), frame)
	
	
	while(1):
		n = cv2.getTrackbarPos('frame',str(args.i)) #get position in the movie
		vidcap.set(cv2.CAP_PROP_POS_MSEC,n/fps*1000) 
		ret,frame = vidcap.read() #read the video at n
		cv2.setMouseCallback(str(args.i), roi_draw) 
		cv2.imshow(str(args.i), frame)
		k=cv2.waitKey()
		if k == 27:
			break	 #quit with ESC
	 
	# close all open windows
	cv2.destroyAllWindows()
			
			
			
############################
#
# -s option Stabilize 
#
############################



if args.s:
	

	#create the output directory for stabilized images, erased everytime.
	dirname = 'output'
	
	owd = os.getcwd()
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	else:
		
		os.chdir(dirname)
		filelist = [ f for f in os.listdir(".") if f.endswith(".jpg") ]
		for f in filelist:
			os.remove(f)
		os.chdir(owd)
	
	

#################################################################################
#
# 	read data from xml
# 		- mandatory: movie, roi, template
# 		- optional: newmovie, enhance, spot, contrast, scalebar, ffmpeg, preview
#
###############################################################################


	tree = ET.parse(args.filename)
	root = tree.getroot()

	for roi in root.findall('roi'):
		
		 xr= int(roi.get('x'))
		 yr= int(roi.get('y'))
		 wr= int(roi.get('w'))
		 hr= int(roi.get('h'))
		 

	for template in root.findall('template'):
		
		 xt= int(template.get('x'))
		 yt= int(template.get('y'))
		 wt= int(template.get('w'))
		 ht= int(template.get('h'))

	for enhance in root.findall('enhance'):
		timeavg=int(enhance.get('denoise'))
		sm=int(enhance.get('smooth'))
		dev=int(enhance.get('deviation'))
		
	for contrast in root.findall('contrast'):
		cl=float(contrast.get('clahelimit'))
		cx=int(contrast.get('clahex'))
		cy=int(contrast.get('clahey'))
		
	for newmovie in root.findall('newmovie'):
		start=int(newmovie.get('start'))
		end=int(newmovie.get('end'))
		step=int(newmovie.get('step'))
		
	for spot in root.findall('spot'):
		inpaint=str(spot.get('inpaint'))
		xs= int(spot.get('x'))
		ys= int(spot.get('y'))
		ws= int(spot.get('w'))
		hs= int(spot.get('h'))
		inp=int(spot.get('inp'))
		istart=int(spot.get('start'))
		iend=int(spot.get('end'))
		blurradius=int(spot.get('radius'))
		
	for scalebar in root.findall('scalebar'):
		mag= int(scalebar.get('mag'))
		size= int(scalebar.get('size'))
	
	for ffmpeg in root.findall('ffmpeg'):
		deinterlace= str(ffmpeg.get('deinterlace'))
		output=str(ffmpeg.get('output'))
	
	for preview in root.findall('preview'):
		prevraw= str(preview.get('raw'))
		prevstab=str(preview.get('stabilize'))
		prevenhance=str(preview.get('enhance'))
		
	for movie in root.findall('movie'):
		filvid=str(movie.get('name'))


###############################
#
# Initialize
#
##############################
	print "============ Run Pystab ================="
	vidcap = cv2.VideoCapture(filvid)
	
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 					#video length
	fps= vidcap.get(cv2.CAP_PROP_FPS) 
	
	ret,frame = vidcap.read() 											#read the video
	rows,cols,ch = frame.shape 											#frame width height and channel numbers
	

	if root.findall('newmovie')==[]:    								#default movie if newmovie not specified
		start=0
		end=length
	
	end_ms=int(end*1000/fps)
	
	image = np.empty(length, dtype=object)        						#store the images
	image_rect = np.empty(length, dtype=object)      
	stabilize = np.empty(length, dtype=object)
	smooth = np.empty(length, dtype=object)
		
	average_spot=np.zeros((rows,cols,ch), np.float64)
	count=0
	ct=0
	
##################################################
#
# Read raw frame
#
##################################################
	print "Read raw video..."
	while(1):
		ret ,frame = vidcap.read()
		if ret == True :
			image[count] =frame
			image_rect[count] =frame
			if root.findall('spot')!=[]:
				if count>istart and count<iend:
					average_spot=cv2.accumulate(image[count],average_spot)
			count+=1
			if root.findall('preview')!=[]:
				if prevraw=='yes':
					cv2.imshow('Raw video',frame)
					k = cv2.waitKey(1) & 0xff
		else:
			break
	print "Done"
##################################################
#
# Spot option (camera imperfection removal)
#
##################################################			
	
	
	if root.findall('spot')!=[]:
		if inpaint!='yes':
			average_spot=cv2.convertScaleAbs(average_spot/(iend-istart)) 	#compute the average image when the camera moves rapidly
			blurred=cv2.blur(average_spot,(blurradius,blurradius)) 			#blur the average image
			background=cv2.subtract(blurred,average_spot) 					#get the background
			cv2.imwrite('background.jpg', background)						#write background.jpg
			
		else:															#mask the spot area if the inpainting technique is used
			mask = np.zeros((rows,cols,ch),np.uint8)
			mask[ys:ys+hs,xs:xs+ws] = image[start][ys:ys+hs,xs:xs+ws]
	
	fixed = image[start][yt:yt+ht,xt:xt+wt]								#crop in the ROI
	S=np.zeros((end-start,2))
	
##################################################
#
# Stabilize
#
##################################################
	print "Run stabilize..."
	
	for i in range(start, end, step):
		if root.findall('spot')!=[]:
			if inpaint!='yes':
				image[i]=cv2.add(image[i],background)						#add the background to the image for spot removal
			else:
				image[i]=cv2.inpaint(image[i],mask[:,:,0],inp,cv2.INPAINT_NS) #or use inpainting
		
		res=cv2.matchTemplate(image[i], fixed, cv2.TM_CCORR_NORMED) 	#Template matching for the fixed point
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 		#Find the position of the template
		top_left = max_loc
		bottom_right = (top_left[0] + wt, top_left[1] + ht)
		f=top_left[0]-xt  												# Compute the template shift (trajectory)
		g=top_left[1]-yt
		S[i,0]=f
		S[i,1]=g
		M = np.float32([[1,0,-f],[0,1,-g]]) 
		dst = cv2.warpAffine(image[i],M,(cols,rows)) 					# Translate the window by the template shift
		stabilize[i]=dst[yr:yr+hr,xr:xr+wr] 							# crop in the roi and display and store the stabilized images
		if root.findall('contrast')!=[]:								#if contrast option make the clahe
				clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(cx,cy))
				stabilize[i]=cv2.cvtColor(stabilize[i], cv2.COLOR_BGR2GRAY)
				stabilize[i]=clahe.apply(stabilize[i])
		if root.findall('scalebar')!=[]:								#draw the scalebar
				l=int(size*616*mag/1005/120)
				tex=str(size)+' nm'
				p1=(wr-l-40,hr-20)
				p2=(wr-40,hr-20)
				cv2.line(stabilize[i], p1,p2, (0, 0, 0), thickness=2)
				retval,baseline=cv2.getTextSize(tex,1,1.2,1)
				cv2.putText(stabilize[i],tex,(p1[0]+int(l/2-retval[0]/2),p1[1]-10),1,1.5,(0,0,0),lineType = cv2.LINE_AA)   
		
		if root.findall('preview')!=[]:
				if prevstab=='yes':
					cv2.imshow('Stabilized',cv2.rectangle(image_rect[i], top_left, bottom_right, (0, 255, 0), 2))							#Show the stabilized video
					cv2.waitKey(1) & 0xff
		if root.findall('enhance')==[]:
			cv2.imwrite(os.path.join('output', "stabilized%d.jpg"%ct) , stabilize[i])	#write the stabilized images in the output folder
			ct+=1
	
	print "Done"
##################################################
#
# Enhance option (smoothing trajectory and denoise)
#
##################################################
	
	if root.findall('enhance')!=[]:
		print "Run Enhance..."
		#S2=np.array(S).reshape(-1,2)
		S2=S
		S3=S[::step,:]
		def meanaveraging(x, N):
			return np.convolve(x, np.ones((N,))/N)[int((N-1)/2):]
			
		S22=np.vstack((S2, S2[-sm*step:,:]))
		fr=np.linspace(0,np.shape(S3)[0]-1,np.shape(S3)[0])
		
		R=step*meanaveraging(S22[:,0],sm*step)									#calculate the running image average according a sliding window of width sm (width in the xml)
		Q=step*meanaveraging(S22[:,1],sm*step)
		plt.plot(fr,S3[:,0], "b-",fr,S3[:,1], "g-", fr, Q[:np.shape(S2)[0]:step], "ro", fr,R[:np.shape(S2)[0]:step], "ro")	#plot the raw and smoothed trajectory and the save it to trajectory.png:
		plt.savefig('trajectory.png')										
		
		for i in range(start,end,step):
			if np.abs(np.abs(R[i-start])-np.abs(S2[i-start,0]))>dev or np.abs(np.abs(Q[i-start])-np.abs(S2[i-start,1]))>dev:	#Do not take into consideration the smoothed trajectory 
					M = np.float32([[1,0,-S2[i-start,0]],[0,1,-S2[i-start,1]]])													#if it deviates significantly from the true one
			else:
					M = np.float32([[1,0,-R[i-start]],[0,1,-Q[i-start]]])
					
								
			dst = cv2.warpAffine(image[i],M,(cols,rows)) 				# Translate the window by the template shift
			
			smooth[i]=dst[yr:yr+hr,xr:xr+wr] 							#crop in the roi
					
		
		
		for i in range(start, end,step):										#denoise by performing a time averaging over 2timeavg images (denoise in the xml)
			cc=0
			average_image=np.zeros(smooth[start].shape, np.float64)
			for j in range(-timeavg*step, timeavg*step, step):
					if i+j>=start and i+j<end-1:
						average_image=cv2.accumulate(smooth[i+j],average_image)
						cc+=1
						
						
						
			average_image= cv2.convertScaleAbs(average_image/cc)
			average_image=cv2.cvtColor(average_image, cv2.COLOR_BGR2GRAY)
			if root.findall('contrast')!=[]:								#if contrast option make the clahe
				clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(cx,cy))
				average_image = clahe.apply(average_image)
			if root.findall('scalebar')!=[]:
				l=int(size*616*mag/1005/120)
				tex=str(size)+' nm'
				p1=(wr-l-40,hr-20)
				p2=(wr-40,hr-20)
				cv2.line(average_image, p1,p2, (0, 0, 0), thickness=2)
				retval,baseline=cv2.getTextSize(tex,1,1.2,1)
				cv2.putText(average_image,tex,(p1[0]+int(l/2-retval[0]/2),p1[1]-10),1,1.5,(0,0,0),lineType = cv2.LINE_AA)   
			ct+=1
			if root.findall('preview')!=[]:
				if prevenhance=='yes':
					cv2.imshow('Enhanced', average_image)
					cv2.waitKey(40)
			cv2.imwrite(os.path.join('output', "stabilized%d.jpg"%ct) , average_image)	#Write smoothed stabilized images in the output directory
			
	print "Done"
			
###########################
#
# ffmpeg option 
#
###########################
		
		
	if root.findall('ffmpeg')!=[]:
		
		print "Exporting video..."
		
		if deinterlace=='yes':
			subprocess.call(['ffmpeg', '-loglevel', 'panic', '-y', '-i', 'output/stabilized%d.jpg', '-vf', 'yadif=0:-1:0', output])
		else:
			subprocess.call(['ffmpeg', '-loglevel', 'panic', '-y', '-i', 'output/stabilized%d.jpg', output])
		print "Done"	
		
#Your statements here
else:
	print "ERROR: see help for syntax"
b = datetime.now()
c=b-a
print "Performed in "+str(c)+" s"
print "========================================"
