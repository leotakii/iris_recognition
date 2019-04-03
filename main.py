#!/usr/bin/python

# Import the required modules
#from utils import flood_fill,bilinear_interpolation
import cv2, sys, os
import cv2.cv as cv
import math as mt
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.linalg as npla
import scipy.misc as spm
import string

# images will contains face images


_pathEye  = './databases/CASIA-IrisV4-Interval'
_pathEye  = './databases/CASIA-Iris-Lamp-100'
_pathMask = './databases/CASIA-IrisV4-Lamp-100-mask'
_pathIris = './databases/CASIA-IrisV4-Lamp-100-iris'
_pathNorm = './databases/CASIA-IrisV4-Lamp-100-norm'

def getEyeImages(pathEye = _pathEye, pathMask = _pathMask, pathIris = _pathIris, pathNorm = _pathNorm):

	EyeImages  = []
	MaskImages = []
	IrisImages = []
	NormImages = []
	# subjets will contains the subject identification number assigned to the image
	idEyeList = []

	if pathMask and not os.path.exists(pathMask):
		os.makedirs(pathMask)
	if pathIris and not os.path.exists(pathIris):
		os.makedirs(pathIris)
	if pathNorm and not os.path.exists(pathNorm):
		os.makedirs(pathNorm)

	subjects_paths = [os.path.join(pathEye, d) for d in os.listdir(pathEye) if os.path.isdir(os.path.join(pathEye,d))]
	for s,subject_paths in enumerate(subjects_paths, start=1):
		# Get the label of the subject
		nsb = int(os.path.split(subject_paths)[1]) 

		# if pathMask and not os.path.exists(os.path.join(pathMask,os.path.split(subject_paths)[1])):
		# 	os.makedirs(os.path.join(pathMask,os.path.split(subject_paths)[1]))
		# if pathIris and not os.path.exists(os.path.join(pathIris,os.path.split(subject_paths)[1])):
		# 	os.makedirs(os.path.join(pathIris,os.path.split(subject_paths)[1]))
		# if pathNorm and not os.path.exists(os.path.join(pathNorm,os.path.split(subject_paths)[1])):
		# 	os.makedirs(os.path.join(pathNorm,os.path.split(subject_paths)[1]))

		side_paths = [os.path.join(subject_paths, d) for d in os.listdir(subject_paths) if os.path.isdir(os.path.join(subject_paths,d))]
		for e,side_path in enumerate(side_paths, start=1):
			idEye = 2*nsb + (-1 if os.path.split(side_path)[1]  == 'L' else 0 )
			print '{0}/{1}:{2}'.format(nsb,idEye,side_path)

			# if pathMask and not os.path.exists(os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
			# 	os.makedirs(os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))
			# if pathIris and not os.path.exists(os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
			# 	os.makedirs(os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))
			# if pathNorm and not os.path.exists(os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
			# 	os.makedirs(os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))

			eye_paths = [os.path.join(side_path, f) for f in os.listdir(side_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(side_path,f)) ]
			for y,eye_path in enumerate(eye_paths,start=1):
				# Read the image and convert to grayscale
				imgEye_pil = Image.open(eye_path).convert('L')
				# Convert the image format into numpy array
				imgEye = np.array(imgEye_pil, 'uint8') # normalization
				EyeImages.append(imgEye)
				idEyeList.append(idEye)

				print '{0}:{1}'.format(y,eye_path)
				sys.stdout.flush()

				imgMask, imgIris = SegIris(imgEye)
				# if pathMask:
				# 	imgpathMask = os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
				# 	cv2.imwrite(imgpathMask,imgMask)
				# if pathIris:
				# 	imgpathIris = os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
				# 	cv2.imwrite(imgpathIris,imgIris)

				# imgNorm = Mask2Norm(imgIris,imgMask,(32,256))
				# if pathNorm:
				# 	imgpathNorm = os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
				# 	cv2.imwrite(imgpathNorm,imgNorm)

	#					fig, aplt = plt.subplots(2,2)
	#					aplt[0,0].imshow(imgEye,cmap='Greys_r')
	#					aplt[0,1].imshow(imgMask,cmap='Greys_r')
	#					aplt[1,0].imshow(imgIris,cmap='Greys_r')
	#					aplt[1,1].imshow(imgNorm,cmap='Greys_r')
	#					plt.pause(_waitingtime)
	#					plt.close()

				IrisImages.append(imgIris)
				MaskImages.append(imgMask)
				NormImages.append(imgNorm)
			print ' done.'

getEyeImages()