#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li #
#  USC ID: 6503378943 #
#  USC Email: zongjian@usc.edu #
#  Submission Date: 28th, Apr 2019 #
####################################

# Modify on the basis of provided FF-CNN code.  ref:
# https://github.com/davidsonic/Interpretable_CNN

from tensorflow.python.platform import flags
import pickle
import numpy as np
import sklearn
import cv2
import keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import os
import imageio
import argparse

import data
import saab_compact as saab
from laws import laws

def getKernel(train_images, train_labels, kernel_sizes, num_kernels, energy_percent, use_num_images, class_list):
	# conv weights
	print("Parameters: ")
	print("Use_classes: ", class_list)
	print("Kernel_sizes: ", kernel_sizes)
	print("Number_kernels: ", num_kernels)
	print("Energy_percent :", energy_percent)
	print("Number_use_images: ", use_num_images)
	pca_params = saab.multi_Saab_transform(train_images, 
										   train_labels,
										   kernel_sizes=kernel_sizes,
										   num_kernels=num_kernels,
										   energy_percent=energy_percent,
										   use_num_images=use_num_images,
										   use_classes=class_list)

	# save data
	#with open('pca_params_compact.pkl', 'wb') as fw:
	#	pickle.dump(pca_params, fw)

	return pca_params

def getFeature(images, pca_params, batch_size):
	# Training
	num_batch = images.shape[0] // batch_size
	feature = np.concatenate([saab.initialize(train_image_batch, pca_params) for train_image_batch in np.array_split(images, num_batch)])
	
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')

	# save data
	#feat = {"feature": feature}
	#with open('feat_compact.pkl', 'wb') as fw:
	#	pickle.dump(feat, fw)
	
	return feature

def getWeight(train_feature, train_labels, factors):
	#import random
	#random.seed(9001)
	num_FC_input_kernels = 15 + 1
	num_final_classes = 10
	output_class_factors = (*factors, 1)

	assert output_class_factors[-1] == 1
	num_train_imgs, num_FC_input_kernels, feature_height, feature_width = train_feature.shape
	train_feature = np.moveaxis(train_feature, 1, 3)
	train_feature = train_feature.reshape(num_train_imgs, -1)
	num_output_classes = [num_final_classes * factor for factor in output_class_factors]
	num_layers = len(output_class_factors)

	# feature normalization
	# std_var = (np.std(feature, axis=0)).reshape(1, -1)
	# feature = feature / std_var

	output_feature = train_feature
	fc_weight = {}
	fc_feature = {}
	for k in range(num_layers):
		input_feature = output_feature
		if k < num_layers - 1:
			# define
			num_input_class = input_feature.shape[1]
			num_output_class = num_output_classes[k]
			def getClassCounts(output_labels):
				class_counts = np.zeros((num_output_class, num_final_classes), dtype=np.float32)
				for i in range(num_train_imgs):
					class_counts[output_labels[i], train_labels[i]] += 1
				return class_counts
			def getAccuracy(class_counts):
				return np.sum(np.amax(class_counts, axis=1)) / num_train_imgs
			dc = np.ones((num_train_imgs, 1), dtype=np.float32)

			# Kmeans
			kmeans = KMeans(n_clusters=num_output_class).fit(input_feature)
			output_labels = kmeans.labels_
			class_counts = getClassCounts(output_labels)
			print("{} layer Kmean (just ref) training acc is {}".format(k, getAccuracy(class_counts)))

			# Compute centroids
			final_label_of_output_classes = np.argmax(class_counts, axis=1)
			centroids = np.zeros((num_output_class, num_input_class), dtype=np.float32)
			for output_class_index in range(num_output_class):
				belonging_feature_sum = np.zeros(num_input_class, dtype=np.float32)
				count = 0
				for i in range(num_train_imgs):
					if output_labels[i] == output_class_index and final_label_of_output_classes[output_class_index] == train_labels[i]:
						belonging_feature_sum += input_feature[i]
						count += 1
				centroids[output_class_index] = belonging_feature_sum / count

			# Compute one hot vector
			expected_output = np.zeros((num_train_imgs, num_output_class), dtype=np.float32)
			for i in range(num_train_imgs):
				if final_label_of_output_classes[output_labels[i]] == train_labels[i]:
					sub_class = output_labels[i]
				else:
					distance_assigned = euclidean_distances(input_feature[i].reshape(1, -1), centroids[output_labels[i]].reshape(1, -1))
					cluster_special = [j for j in range(num_output_class) if final_label_of_output_classes[j] == train_labels[i]]
					distance = np.zeros(len(cluster_special))
					for j in range(len(cluster_special)):
						distance[j] = euclidean_distances(input_feature[i].reshape(1, -1), centroids[cluster_special[j]].reshape(1, -1))
					sub_class = cluster_special[np.argmin(distance)]
				expected_output[i, sub_class] = 1

			# least square regression
			input_feature_with_dc = np.concatenate((dc, input_feature), axis=1)
			layer_weights = np.matmul(np.linalg.pinv(input_feature_with_dc), expected_output)
			output_feature = np.matmul(input_feature_with_dc, layer_weights) # calc true output
			output_labels = np.argmax(output_feature, axis=1)
			print("{} layer LSR training acc is {}".format(k, getAccuracy(getClassCounts(output_labels))))

			# Relu
			output_feature = np.maximum(output_feature, 0)

		else:
			# least square regression
			expected_output = keras.utils.to_categorical(train_labels, 10)
			input_feature_with_dc = np.concatenate((dc, input_feature), axis=1)
			layer_weights = np.matmul(np.linalg.pinv(input_feature_with_dc), expected_output).astype(np.float32)
			output_feature = np.matmul(input_feature_with_dc, layer_weights)
		fc_weight[k] = layer_weights
		fc_feature[k] = output_feature
		print(k, ' layer LSR weight shape:', layer_weights.shape)
		print(k, ' layer LSR output shape:', output_feature.shape)

	output_labels = np.argmax(output_feature, axis=1)
	acc_train = sklearn.metrics.accuracy_score(train_labels, output_labels)
	print("Final training acc is {}".format(acc_train))

	# save data
	#with open('llsr_weights_compact_v2.pkl', 'wb') as fw:
	#	pickle.dump(weights, fw, protocol=2)
	#with open('llsr_bias_compact_v2.pkl', 'wb') as fw:
	#	pickle.dump(bias, fw, protocol=2)

	return fc_weight, fc_feature

def main():
	# args
	parser = argparse.ArgumentParser(description = "For USC EE569 2019 spring home work 6 by Zongjian Li.")
	parser.add_argument("-c", "--use_classes", default="0-9", help="Supported format: 0,1,5-9")
	parser.add_argument("-l", "--laws_filter_name", default=None, type=str, help="Laws filter, default is None. Fomat'L3L3'")
	parser.add_argument("-s", "--kernel_sizes", default="5,5", help="Kernels size for each stage. Format: '3,3'")
	parser.add_argument("-k", "--num_kernels", default="5,15", help="Num of kernels for each stage. Format: '4,10'")
	parser.add_argument("-f", "--fc_factors", default="12,8", help="Factor of num of kernels for each stage in FC. Format: '12,8'")
	parser.add_argument("-e", "--energy_percent", default=None, type=float, help="Energy to be preserved in each stage")
	parser.add_argument("-n", "--num_conv_images", default=-1, type=int, help="Num of images used for conv training")
	parser.add_argument("-b", "--batch_size", default=10000, type=int, help="batch size")
	parser.add_argument("-o", "--save_filename", default="ff-cnn.w", help="save filename")
	args = parser.parse_args()
	args.kernel_sizes = saab.parse_list_string(args.kernel_sizes)
	args.num_kernels = saab.parse_list_string(args.num_kernels) if args.num_kernels else None
	args.fc_factors = saab.parse_list_string(args.fc_factors)

	# load training data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data(args.use_classes)
	print("Training image size:", train_images.shape)
	print("Testing_image size:", test_images.shape)
	print("Training images.dtype ", train_images.dtype)
	train_images = np.moveaxis(train_images, 3, 1)
	test_images = np.moveaxis(test_images, 3, 1)

	train_images = laws(train_images, args.laws_filter_name)
	conv_weight = getKernel(train_images, train_labels, args.kernel_sizes, args.num_kernels, args.energy_percent, args.num_conv_images, class_list)
	conv_feature = getFeature(train_images, conv_weight, args.batch_size)
	fc_weight, fc_feature = getWeight(conv_feature, train_labels, args.fc_factors)

	d = {"conv_weight": conv_weight, "fc_weight": fc_weight}
	if not args.laws_filter_name is None:
		d["laws_filter_name"] = args.laws_filter_name
	#d["conv_feature"] = conv_feature
	#d["fc_feature"] = fc_feature
	with open(args.save_filename, "wb") as f:
		pickle.dump(d, f)

if __name__ == "__main__":
	main()
