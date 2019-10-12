#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li #
#  USC ID: 6503378943 #
#  USC Email: zongjian@usc.edu #
#  Submission Date: 28th, Apr 2019 #
####################################

import pickle
import numpy as np
import sklearn
import argparse
import os
import data
import saab_compact as saab
from laws import laws

DEFAULT_BATCH_SIZE = 10000

def feedforward(images, ffcnn_params, batch_size=DEFAULT_BATCH_SIZE):
	# prepare
	num_imgs = images.shape[0]
	conv_weight = ffcnn_params["conv_weight"]
	fc_weight = ffcnn_params["fc_weight"]
	#train_conv_feature = ffcnn_params["conv_feature"]
	#train_fc_feature = ffcnn_params["fc_feature"]
	num_batch = num_imgs // batch_size
	dc = np.ones((num_imgs, 1), dtype=np.float32)

	# calc conv feature
	output_feature = np.concatenate([saab.initialize(train_image_batch, conv_weight) for train_image_batch in np.array_split(images, num_batch)])
	#assert num_imgs != train_conv_feature.shape[0] or np.array_equal(train_conv_feature, output_feature.reshape(num_imgs, -1))

	# calc 
	output_feature = np.moveaxis(output_feature, 1, 3)
	output_feature = output_feature.reshape(num_imgs, -1)
	num_layers = len(fc_weight)
	for layer in range(num_layers):
		input_feature = output_feature
		layer_weight = fc_weight[layer]
		input_feature_with_dc = np.concatenate((dc, input_feature), axis=1)
		output_feature = np.matmul(input_feature_with_dc, layer_weight) 
		if layer < num_layers - 1:
			output_feature = np.maximum(output_feature, 0)
		#train_layer_feature = train_fc_feature[layer]
		#assert num_imgs != train_layer_feature.shape[0] or np.array_equal(train_layer_feature, output_feature)

	return output_feature

def acc(true_labels, pred_labels):
	return sklearn.metrics.accuracy_score(true_labels, pred_labels)

def main():
	# args
	parser = argparse.ArgumentParser(description = "For USC EE569 2019 spring home work 6 by Zongjian Li.")
	parser.add_argument("-i", "--input_folder", default=".", help="Intput FF-CNN weight file folder.")
	parser.add_argument("-o", "--output_filename", default="all-ff-cnn-output.l", help="Saved return lable filename.")
	parser.add_argument("-e", "--extension", default=".w", help="Intput FF-CNN weight filename extension.")
	parser.add_argument("-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size of feature gen.")
	args = parser.parse_args()

	#load data
	train_images, train_labels, test_images, test_labels, _ = data.import_data("0-9")
	train_images = np.moveaxis(train_images, 3, 1)
	test_images = np.moveaxis(test_images, 3, 1)
	
	filenames = [(name, "{}/{}".format(args.input_folder, name)) for name in os.listdir(args.input_folder)]
	filenames = [(name, full_name) for name, full_name in filenames if os.path.isfile(full_name) and full_name.endswith(args.extension)]
	all_returned_labels = {}
	for name, full_name in filenames:
		#load ff-cnn
		with open(full_name, "rb") as f:
			params = pickle.load(f)
		# laws filter
		temp_train_images = laws(train_images, params["laws_filter_name"]) if "laws_filter_name" in params else train_images
		temp_test_images = laws(test_images, params["laws_filter_name"]) if "laws_filter_name" in params else test_images
		# calc acc
		train_returned_feature = feedforward(temp_train_images, params, batch_size=args.batch_size)
		train_returned_labels = np.argmax(train_returned_feature, axis=1) 
		train_acc = acc(train_labels, train_returned_labels)
		test_returned_feature = feedforward(temp_test_images, params, batch_size=args.batch_size)
		test_returned_labels = np.argmax(test_returned_feature, axis=1) 
		test_acc = acc(test_labels, test_returned_labels)
		# record
		all_returned_labels[name] = {"train_feature": train_returned_feature, "train_label": train_returned_labels, "train_acc": train_acc, "test_feature": test_returned_feature, "test_label": test_returned_labels, "test_acc": test_acc}

	# print result
	for name, d in all_returned_labels.items():
		print("{} Train acc is {}".format(name, d["train_acc"]))
		print("{} Test acc is {}".format(name, d["test_acc"]))
	# write data
	with open(args.output_filename, "wb") as f:
		pickle.dump(all_returned_labels, f)

if __name__ == "__main__":
	main()