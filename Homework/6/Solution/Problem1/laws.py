#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li #
#  USC ID: 6503378943 #
#  USC Email: zongjian@usc.edu #
#  Submission Date: 28th, Apr 2019 #
####################################

import numpy as np
import cv2

def laws(images, laws_filter_name):
	if laws_filter_name is None:
		return images
	laws_filter_name = laws_filter_name.upper()
	assert len(laws_filter_name) == 4
	laws_1D = {
		"L3": np.array((1, 2, 1)).reshape(1, -1),
		"E3": np.array((-1, 0, 1)).reshape(1, -1),
		"S3": np.array((-1, 2, -1)).reshape(1, -1),
		}
	vert_filter = laws_1D[laws_filter_name[:2]]
	hori_filter= laws_1D[laws_filter_name[2:]]
	filter = np.matmul(hori_filter.T, vert_filter)
	images = images.reshape(-1, images.shape[-2], images.shape[-1]) # hack stupid FF-CNN github code!
	images = np.array([cv2.filter2D(img, -1, filter) for img in images])
	images = images.reshape(-1, 1, images.shape[-2], images.shape[-1]) # hack stupid FF-CNN github code!
	return images
