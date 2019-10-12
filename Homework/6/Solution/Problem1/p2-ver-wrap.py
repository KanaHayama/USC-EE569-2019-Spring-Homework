#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li               #
#  USC ID: 6503378943              #
#  USC Email: zongjian@usc.edu     #
#  Submission Date: 28th, Apr 2019 #
####################################

# Modify on the basis of provided FF-CNN code. ref: https://github.com/davidsonic/Interpretable_CNN

from tensorflow.python.platform import flags
import data
import saab_compact as saab
import numpy as np
import os
import imageio

def main():
	# args
	flags.DEFINE_string("output_path", None, "The output dir to save params")
	flags.DEFINE_string("use_classes", "0-9", "Supported format: 0,1,5-9")
	flags.DEFINE_string("kernel_sizes", "4,4", "Kernels size for each stage. Format: '3,3'")
	flags.DEFINE_string("num_kernels", "5,15", "Num of kernels for each stage. Format: '4,10'")
	flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
	flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
	flags.DEFINE_boolean("overlapping", False, "Overlapping windows")
	flags.DEFINE_boolean("maxpooling", False, "Maxpooling")
	flags.DEFINE_string("image_folder", ".", "Folder of 4 testing images")
	flags.DEFINE_string("recover_suffix", "_recover", "Suffix of recovered images")
	args = flags.FLAGS
	assert args.overlapping == False
	assert args.maxpooling == False

	# load training data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data(args.use_classes)
	print("Training image size:", train_images.shape)
	print("Testing_image size:", test_images.shape)
	print("Training images.dtype ", train_images.dtype)
	train_images = np.moveaxis(train_images, 3, 1)
	kernel_sizes = saab.parse_list_string(args.kernel_sizes)
	if args.num_kernels:
		num_kernels = saab.parse_list_string(args.num_kernels)
	else:
		num_kernels = None
	print("Parameters: ")
	print("Use_classes: ", class_list)
	print("Kernel_sizes: ", kernel_sizes)
	print("Number_kernels: ", num_kernels)
	print("Energy_percent :", args.energy_percent)
	print("Number_use_images: ", args.use_num_images)
	print("Overlapping: ", args.overlapping)
	print("Max_pooling: ", args.maxpooling)
	
	# conv weights
	pca_params = saab.multi_Saab_transform(train_images, 
										   train_labels,
										   kernel_sizes=kernel_sizes,
										   num_kernels=num_kernels,
										   energy_percent=args.energy_percent,
										   use_num_images=args.use_num_images,
										   use_classes=class_list,
										   overlapping=args.overlapping,
										   maxpooling=args.maxpooling)

	# load 4 images
	DEFAULT_SIZE = (32, 32)
	imgs_byte = []
	filenames = ["{}/{}".format(args.image_folder, name) for name in os.listdir(args.image_folder)]
	png_filenames = [name for name in filenames if os.path.isfile(name) and name.endswith(".png") and not name.endswith(args.recover_suffix + ".png")]
	filenames = []
	for img_name in png_filenames:
		img = imageio.imread(img_name)
		if img.shape == DEFAULT_SIZE:
			imgs_byte.append(img)
			filenames.append(img_name)
	assert len(imgs_byte) > 0
	imgs_byte = np.stack(imgs_byte)
	imgs = imgs_byte / np.iinfo(np.uint8).max
	imgs = np.float32(imgs)
	imgs = imgs.reshape(-1, imgs.shape[-2], imgs.shape[-1], 1)
	imgs = np.moveaxis(imgs, 3, 1)

	# calc features
	feature = saab.initialize(imgs, 
							  pca_params, 
							  overlapping=args.overlapping, 
							  maxpooling=args.maxpooling)
	#feature = feature.reshape(feature.shape[0], -1)
	print("S4 shape:", feature.shape)

	# recover
	## ①矩阵乘法逆运算 ②saab.initialize里没把dc分出来，dc到底要不要分出来处理，怎么处理，为什么train和test的流程不一样，discussion里那个+bias后16D→10D后加DC怎么体现的 ③pca_mean有什么用，feature_expectation有什么用
	num_layers = pca_params['num_layers']
	kernel_sizes = pca_params['kernel_size']
	sample_images = feature
	for i in range(num_layers - 1, -1, -1):
		# read params
		feature_expectation = pca_params['Layer_%d/feature_expectation' % i].astype(np.float32)
		kernels = pca_params['Layer_%d/kernel' % i].astype(np.float32)
		mean = pca_params['Layer_%d/pca_mean' % i].astype(np.float32) #有啥用？
		c = sample_images.shape[1]
		h = sample_images.shape[2]
		w = sample_images.shape[3]
		# 
		sample_images = np.moveaxis(sample_images, 1, 3)
		transformed = sample_images.reshape(-1, sample_images.shape[-1])
		if i == 0:
			sample_patches_centered = np.matmul(transformed, np.linalg.pinv(np.transpose(kernels)))
		else:
			bias = pca_params['Layer_%d/bias' % i].astype(np.float32)
			e = np.zeros((1, kernels.shape[0]), dtype=np.float32)
			e[0, 0] = 1
			transformed += bias * e
			sample_patches_centered_w_bias = np.matmul(transformed, np.linalg.pinv(np.transpose(kernels)))
			num_channels = kernels.shape[1]
			sample_patches_centered = sample_patches_centered_w_bias - 1 / np.sqrt(num_channels) * bias
		sample_patches = sample_patches_centered + feature_expectation
		prev_num_kernel = (0 if i == 0 else num_kernels[i - 1]) + 1
		sample_patches = sample_patches.reshape(-1, h, w, prev_num_kernel, kernel_sizes[0], kernel_sizes[1])
		# copy data
		window_stride = 1 if args.overlapping else kernel_sizes[i]
		sample_images = np.zeros((sample_patches.shape[0], prev_num_kernel, (h - 1) * window_stride +  kernel_sizes[0], (w - 1) * window_stride +  kernel_sizes[1]))
		for l in range(sample_patches.shape[0]):
			for i in range(h):
				for j in range(w):
					for k in range(prev_num_kernel):
						h_start = i * window_stride
						h_end = h_start + sample_patches.shape[-2]
						w_start = j * window_stride
						w_end = w_start + sample_patches.shape[-1]
						sample_images[l, k, h_start : h_end, w_start : w_end] = sample_patches[l, i, j, k, :, :]
	res_imgs = sample_images.reshape(-1, sample_images.shape[-2], sample_images.shape[-1])
	res_imgs = np.clip(res_imgs, 0, 1)
	
	# save 4 images
	res_imgs_byte = res_imgs * np.iinfo(np.uint8).max
	res_imgs_byte = np.uint8(res_imgs_byte)
	for res_img, filename in zip(res_imgs_byte, filenames):
		filename = ("{}" + args.recover_suffix + "{}").format(*os.path.splitext(filename))
		imageio.imwrite(filename, res_img)

	# PSNR
	for filename, img, res_img in zip(filenames, imgs, res_imgs):
		mse = np.mean((img - res_img) ** 2)
		assert mse != 0
		psnr = 10 * 2 * np.log10(1 / np.sqrt(mse))
		print("PSNR of \"{}\" is {}".format(filename, psnr))

if __name__ == "__main__":
	main()
