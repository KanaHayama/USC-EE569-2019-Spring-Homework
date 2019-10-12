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
import imageio
import keras
import data
import saab_compact as saab

def acc(true_labels, pred_labels):
	return sklearn.metrics.accuracy_score(true_labels, pred_labels)

def main():
	# args
	parser = argparse.ArgumentParser(description = "For USC EE569 2019 spring home work 6 by Zongjian Li.")
	parser.add_argument("-i", "--input_filename", default="all-ff-cnn-output.l", help="Saved return lable filename.")
	parser.add_argument("-o", "--output_folder", default="./wrong_predicts/", help="Outpuf folder of wrong predict images.")
	parser.add_argument("-m", "--bp_model_filename", default="trained_model", help="Trained LeNet-5 keras model filename.")
	parser.add_argument("-n", "--num_components", default=None, type=int, help="# of PCA components.")
	args = parser.parse_args()

	#load data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")

	# read data
	with open(args.input_filename, "rb") as f:
		all_returned_labels = pickle.load(f)
	train_feature = np.concatenate([d["train_feature"] for name, d in all_returned_labels.items()], axis=1)
	test_feature = np.concatenate([d["test_feature"] for name, d in all_returned_labels.items()], axis=1)

	# train PCA
	if not args.num_components is None:
		print("Reduce dim to {}".format(args.num_components))
		pca = sklearn.decomposition.PCA(n_components=args.num_components, svd_solver='full')
		train_feature = pca.fit_transform(train_feature)
		test_feature = pca.transform(test_feature)

	# train SVM
	clf = sklearn.svm.SVC()
	clf.fit(train_feature, train_labels) 

	# test SVM
	pred_train_label = clf.predict(train_feature)
	print("Ensemble train acc is {}".format(acc(train_labels, pred_train_label)))
	pred_test_label = clf.predict(test_feature)
	print("Ensemble test acc is {}".format(acc(test_labels, pred_test_label)))

	# load and pred LeNet-5 model
	_, (x_test, y_test) = keras.datasets.mnist.load_data()
	pad_width = (32 - 28) // 2
	x_test = np.pad(x_test, ((0, ), (pad_width, ), (pad_width, )), "edge")
	x_test = np.expand_dims(x_test, axis=3)
	bp_model = keras.models.load_model(args.bp_model_filename)
	bp_pred_test_label = np.argmax(bp_model.predict(x_test), axis=1)

	# find mismatch samples
	test_wrong_idx = np.nonzero(np.not_equal(pred_test_label, test_labels))
	bp_wrong_idx = np.nonzero(np.not_equal(bp_pred_test_label, test_labels))

	same_error = np.intersect1d(test_wrong_idx, bp_wrong_idx)
	only_ff_error = np.setdiff1d(test_wrong_idx, bp_wrong_idx)
	only_bp_error = np.setdiff1d(bp_wrong_idx, test_wrong_idx)

	os.makedirs(args.output_folder, exist_ok=True)
	for name, idx in (("same", same_error), ("only_ff", only_ff_error), ("only_bp", only_bp_error)):
		imgs = np.squeeze(test_images[idx])
		dirname = "{}/{}".format(args.output_folder, name)
		os.makedirs(dirname, exist_ok=True)
		count = 0
		for img in imgs:
			filename = "{}/{:0>4d}.png".format(dirname, count)
			count += 1
			imageio.imwrite(filename, np.uint8(np.clip(img * np.iinfo(np.uint8).max, 0, np.iinfo(np.uint8).max)))

if __name__ == "__main__":
	main()