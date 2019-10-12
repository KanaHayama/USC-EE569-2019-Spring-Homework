#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li               #
#  USC ID: 6503378943              #
#  USC Email: zongjian@usc.edu     #
#  Submission Date: 28th, Apr 2019 #
####################################

import numpy as np
import os
import imageio
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA
import argparse

import data
import saab_compact as saab

KERNEL_SIZE = 4
NUM_KERNELS = (10, 60)
EXTENSION = ".png"
RECOVER_SUFFIX = "_recover"
DEFAULT_IMG_SIZE = (32, 32)

def main():
	parser = argparse.ArgumentParser(description = "For USC EE569 2019 spring home work 6 by Zongjian Li.")
	parser.add_argument("-i", "--image_folder", default=".", help="Sample image file folder.")
	parser.add_argument("-n", "--num_conv_images", default=-1, type=int, help="Num of images used for conv training")
	args = parser.parse_args()
	
	# load training data
	train_images, train_labels, _, _, class_list = data.import_data("0-9")
	train_images = np.moveaxis(train_images, 3, 1)

	# trunc training data
	assert train_images.shape[0] >= args.num_conv_images
	train_images, _ = saab.select_balanced_subset(train_images, train_labels, args.num_conv_images, class_list)
	train_images = np.float32(train_images)

	# train conv params
	weight = {}
	for layer in range(len(NUM_KERNELS)):
		# patches
		num_train_imgs, num_kernel, img_height, img_width= train_images.shape
		feature_size = num_kernel * KERNEL_SIZE * KERNEL_SIZE
		patches = view_as_windows(train_images, (1, num_kernel, KERNEL_SIZE, KERNEL_SIZE), step=(1, num_kernel, KERNEL_SIZE, KERNEL_SIZE))
		patches = patches.reshape(-1, feature_size)
		# gen DC weights
		dc_weights = 1 / np.sqrt(feature_size) * np.ones((1, feature_size), dtype=np.float32)
		# calc DC
		dc = np.matmul(patches, dc_weights.T)
		assert dc.shape[0] == patches.shape[0]
		# remove DC
		patches -= dc
		# calc AC weights
		pca = PCA(n_components=NUM_KERNELS[layer], svd_solver='full')
		pca.fit(patches)
		ac_weights = pca.components_
		pca_mean = pca.mean_
		# calc AC
		ac = np.matmul(patches - pca_mean, ac_weights.T)
		# assemble kernels
		patches = np.concatenate((dc, ac), axis=1)
		# calc bias
		bias = -np.min(patches)
		# add bias
		patches += bias
		# assemble patches
		output_height = img_height // KERNEL_SIZE
		output_width = img_width // KERNEL_SIZE
		train_images = patches.reshape(num_train_imgs, output_height, output_width, -1)
		train_images = np.moveaxis(train_images, 3, 1)
		# record weights
		weight[layer] = {"dc": dc_weights, "ac": ac_weights, "mean":pca_mean, "bias": bias}

	# load 4 images
	imgs_byte = []
	filenames = ["{}/{}".format(args.image_folder, name) for name in os.listdir(IMG_FOLDER)]
	png_filenames = [name for name in filenames if os.path.isfile(name) and name.endswith(EXTENSION) and not name.endswith(RECOVER_SUFFIX + EXTENSION)]
	filenames = []
	for img_name in png_filenames:
		img = imageio.imread(img_name)
		if img.shape == DEFAULT_IMG_SIZE:
			imgs_byte.append(img)
			filenames.append(img_name)
	assert len(imgs_byte) > 0
	imgs_byte = np.stack(imgs_byte)
	imgs = imgs_byte / np.iinfo(np.uint8).max
	imgs = np.float32(imgs)
	imgs = imgs.reshape(-1, imgs.shape[-2], imgs.shape[-1], 1)
	imgs = np.moveaxis(imgs, 3, 1)

	# calc features
	sample_imgs = imgs
	for layer in range(len(NUM_KERNELS)):
		layer_params = weight[layer]
		num_sample_imgs = sample_imgs.shape[0]
		#patches
		_, num_kernel, img_height, img_width= sample_imgs.shape
		feature_size = num_kernel * KERNEL_SIZE * KERNEL_SIZE
		patches = view_as_windows(sample_imgs, (1, num_kernel, KERNEL_SIZE, KERNEL_SIZE), step=(1, num_kernel, KERNEL_SIZE, KERNEL_SIZE))
		patches = patches.reshape(-1, feature_size)
		# calc DC
		dc_weights = layer_params["dc"]
		dc = np.matmul(patches, dc_weights.T)
		# remove DC
		patches -= dc
		# calc AC
		ac_weights = layer_params["ac"]
		pca_mean = layer_params["mean"]
		ac = np.matmul(patches - pca_mean, ac_weights.T)
		# assemble kernels
		patches = np.concatenate((dc, ac), axis=1)
		# add bias
		bias = layer_params["bias"]
		patches += bias
		# assemble patches
		output_height = img_height // KERNEL_SIZE
		output_width = img_width // KERNEL_SIZE
		sample_imgs = patches.reshape(num_sample_imgs, output_height, output_width, -1)
		sample_imgs = np.moveaxis(sample_imgs, 3, 1)

	# reconstruct
	for layer in range(len(NUM_KERNELS) - 1, -1, -1):
		layer_params = weight[layer]
		# assemble patches
		num_sample_imgs, num_kernel, output_height, output_width = sample_imgs.shape
		batch_size = num_sample_imgs * output_height * output_width
		sample_imgs = np.moveaxis(sample_imgs, 1, 3)
		patches = sample_imgs.reshape(batch_size, -1)
		# remove bias
		bias = layer_params["bias"]
		patches -= bias
		# seperate kernels
		dc, ac = np.split(patches, (1, ), axis=1)
		# reverse AC
		ac_weights = layer_params["ac"]
		pca_mean = layer_params["mean"]
		patches = np.matmul(ac, ac_weights) + pca_mean
		# add DC
		patches += dc
		# assemble patches
		prev_num_kernel = (0 if layer == 0 else NUM_KERNELS[layer - 1]) + 1
		window_stride = KERNEL_SIZE
		patches = patches.reshape(num_sample_imgs, output_height, output_width, prev_num_kernel, KERNEL_SIZE, KERNEL_SIZE)
		sample_imgs = np.zeros((patches.shape[0], prev_num_kernel, (output_height - 1) * window_stride +  KERNEL_SIZE, (output_width - 1) * window_stride +  KERNEL_SIZE))
		for l in range(patches.shape[0]):
			for i in range(output_height):
				for j in range(output_width):
					for k in range(prev_num_kernel):
						h_start = i * window_stride
						h_end = h_start + patches.shape[-2]
						w_start = j * window_stride
						w_end = w_start + patches.shape[-1]
						sample_imgs[l, k, h_start : h_end, w_start : w_end] = patches[l, i, j, k, :, :]
		pass

	res_imgs = sample_imgs.reshape(-1, sample_imgs.shape[-2], sample_imgs.shape[-1])
	res_imgs = np.clip(res_imgs, 0, 1)

	# save 4 images
	res_imgs_byte = res_imgs * np.iinfo(np.uint8).max
	res_imgs_byte = np.uint8(res_imgs_byte)
	for res_img, filename in zip(res_imgs_byte, filenames):
		filename = ("{}" + RECOVER_SUFFIX + "{}").format(*os.path.splitext(filename))
		imageio.imwrite(filename, res_img)

	# PSNR
	for filename, img, res_img in zip(filenames, imgs, res_imgs):
		mse = np.mean((img - res_img) ** 2)
		assert mse != 0
		psnr = 10 * 2 * np.log10(1 / np.sqrt(mse))
		print("PSNR of \"{}\" is {}".format(filename, psnr))

if __name__ == "__main__":
	main()

