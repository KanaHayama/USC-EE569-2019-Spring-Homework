#!/usr/bin/env python3
# coding=utf-8

####################################
#  Name: Zongjian Li               #
#  USC ID: 6503378943              #
#  USC Email: zongjian@usc.edu     #
#  Submission Date: 19th,Mar 2019  #
####################################

###################################
#                                 #
#              util               #
#                                 #
###################################
import sys
import os
import numpy as np
import cv2

GRAY_CHANNEL = 1
COLOR_CHANNEL = 3

def readRaw(filename, height, width, channel):
    assert(channel == COLOR_CHANNEL or channel == GRAY_CHANNEL)
    data = np.fromfile(filename, dtype="uint8")
    data = data.reshape(height, width, channel)
    if (channel == 3):
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data

def writeRaw(filename, image):
    pass

###################################
#                                 #
#                b                #
#                                 #
###################################

NOT_DRAW_SINGLE_POINTS = 2


def findLargestScaleKeyPoint(keypoints, descriptors):
    maxScale = None
    result = None
    for i in range(len(keypoints)):
        keypoint = keypoints[i]
        descriptor = descriptors[i]
        scale = dist = np.linalg.norm(descriptor) # definition of scale (in the disscusion tool)
        if maxScale is None or scale > maxScale:
            maxScale = scale
            result = keypoint
    return result

def SIFT(img1, img2):
    grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints1, descriptors1 = sift.detectAndCompute(grayImg1, None)
    keyPoints2, descriptors2 = sift.detectAndCompute(grayImg2, None)

    keyPointImg1 = cv2.drawKeypoints(img1, keyPoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keyPointImg2 = cv2.drawKeypoints(img2, keyPoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("River key points", np.hstack((keyPointImg1, keyPointImg2)))
    bf = cv2.BFMatcher_create(cv2.NORM_L2, True)
    matches = bf.match(descriptors1, descriptors2)
    matches.sort(key=lambda x: x.distance)

    ## best matching
    #matchImg = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, matches[:1], None, flags=NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow("Best matching", matchImg)
    #nearestMatch = matches[0]
    #nearestKeyPointInImg1 = keyPoints1[nearestMatch.queryIdx]
    #nearestKeyPointInImg2 = keyPoints2[nearestMatch.trainIdx]
    #print("Best matching key point in river 1: [pt=%s, size=%f, angle=%f, response=%f, octacv=%d]" % (nearestKeyPointInImg1.pt, nearestKeyPointInImg1.size, nearestKeyPointInImg1.angle, nearestKeyPointInImg1.response, nearestKeyPointInImg1.octave))
    #print("Best matching key point in river 2: [pt=%s, size=%f, angle=%f, response=%f, octacv=%d]" % (nearestKeyPointInImg2.pt, nearestKeyPointInImg2.size, nearestKeyPointInImg2.angle, nearestKeyPointInImg2.response, nearestKeyPointInImg2.octave))

    # answer question
    largestScaleKeyPointInImg1 = findLargestScaleKeyPoint(keyPoints1, descriptors1)
    nearestMatch = [match for match in matches if keyPoints1[match.queryIdx] == largestScaleKeyPointInImg1][0]
    nearestKeyPointInImg2 = keyPoints2[nearestMatch.trainIdx]
    matchImg = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, [nearestMatch], None, matchColor=(255, 0 ,255), flags=NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Question matching", matchImg)
    print("Largest scale key point in river 1: [pt=%s, size=%f, angle=%f, response=%f, octacv=%d]" % (largestScaleKeyPointInImg1.pt, largestScaleKeyPointInImg1.size, largestScaleKeyPointInImg1.angle, largestScaleKeyPointInImg1.response, largestScaleKeyPointInImg1.octave))
    print("Nearest key point in river 2: [pt=%s, size=%f, angle=%f, response=%f, octacv=%d]" % (nearestKeyPointInImg2.pt, nearestKeyPointInImg2.size, nearestKeyPointInImg2.angle, nearestKeyPointInImg2.response, nearestKeyPointInImg2.octave))

###################################
#                                 #
#                c                #
#                                 #
###################################

KMEANS_CLUSTER_NUMBER = 2

def extractFeatures(image, sift):
    keypoints = sift.detect(image, None)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    assert len(keypoints) > 0, "No key point"
    return (keypoints, descriptors)

def combineDescriptors(array):
    return np.vstack(array)

def findClusters(descriptors):
    _, labels, centers = cv2.kmeans(descriptors, KMEANS_CLUSTER_NUMBER, None, ( cv2.TERM_CRITERIA_EPS, 0, sys.float_info.epsilon), 1000, cv2.KMEANS_PP_CENTERS)
    return centers

def cluster(descriptor, clusterCenters):
    result = None
    minDistance = sys.float_info.max
    for center in clusterCenters:
        dist = np.linalg.norm(descriptor - center)
        if dist < minDistance:
            minDistance = dist
            result = center
    return result

def histogram(descriptors, clusterCenters):
    result = [0 for _ in range(clusterCenters.shape[0])]
    for descriptor in descriptors:
        center = cluster(descriptor, clusterCenters)
        index = np.where(np.all(clusterCenters == center, axis=1))
        index = index[0][0]
        result[index] = result[index] + 1
    result = [occurance / len(descriptors) for occurance in result]
    return result

def findNearestHistgram(histgrams, histgram):
    minError = None
    result = None
    for hist in histgrams:
        error = 0
        for i in range(len(histgram)):
            error = error + abs(hist[i] - histgram[i])
        if minError is None or error < minError:
            minError = error
            result = hist
    return result

def bagOfVisualWords(dataImages, sampleImage, scale):
    # scale up
    print("All images scale up to %.3fx" % scale)
    scaledDataImages = [cv2.resize(image, (0, 0), fx=scale, fy=scale) for image in dataImages]
    scaledSampleImage = cv2.resize(sampleImage, (0, 0), fx=scale, fy=scale)

    # gen clusters
    sift = cv2.xfeatures2d.SIFT_create()
    features = [extractFeatures(image, sift) for image in scaledDataImages]
    keyPoints = [keypoints for keypoints, _ in features]
    descriptors = [descriptors for _, descriptors in features]
    allDescriptors = combineDescriptors(descriptors)
    clusterCenters = findClusters(allDescriptors)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("Cluster 1 center: %s" % clusterCenters[0])
    print("Cluster 2 center: %s" % clusterCenters[1])

    # partition
    histgrams = [histogram(descriptors[i], clusterCenters) for i in range(len(scaledDataImages))]
    print("Data images' histgrams: %s" % histgrams)

    # sample feature extraction
    sampleFeature = extractFeatures(scaledSampleImage, sift)
    sampleHistgram = histogram(sampleFeature[1], clusterCenters)
    print("Sample image's histgram: %s" % sampleHistgram)

    # show key points
    keyPointImgs = [cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for img, kp in zip(scaledDataImages, keyPoints)]
    keyPointImgs.append(cv2.drawKeypoints(scaledSampleImage, sampleFeature[0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    cv2.imshow("Digit key points", np.hstack(keyPointImgs))

    # find nearest
    nearestHistgram = findNearestHistgram(histgrams, sampleHistgram)
    print("Nearest histgram: %s" % nearestHistgram)
    nearestIndex = histgrams.index(nearestHistgram)
    return dataImages[nearestIndex]

###################################
#                                 #
#              const              #
#                                 #
###################################

RIVER_HEIGHT = 1024
RIVER_WIDTH = 768
RIVER_CHANNEL = COLOR_CHANNEL
RIVER_1_FILENAME = "river1.raw"
RIVER_2_FILENAME = "river2.raw"

DIGIT_HEIGHT = 28
DIGIT_WIDTH = 28
DIGIT_CHANNEL = GRAY_CHANNEL
DIGIT_DATA_FILENAMES = ("one_1.raw", "one_2.raw", "one_3.raw", "one_4.raw", "one_5.raw", "zero_1.raw", "zero_2.raw", "zero_3.raw", "zero_4.raw", "zero_5.raw", )
DIGIT_SAMPLE_FILENAME = "eight.raw"

DEFAULT_SCALE_FACTOR = 2

###################################
#                                 #
#              main               #
#                                 #
###################################

import argparse

def main():
    # parse args
    parser = argparse.ArgumentParser(description = "For USC EE569 2019 spring home work 4 problem 2 by Zongjian Li.")
    parser.add_argument('--subproblem', '-p', choices = ["b", "c"], default="b", help="Choose sub-problem b or c")
    parser.add_argument('--scale', '-s', metavar="f", type = float, default=DEFAULT_SCALE_FACTOR, help="Scale factor for sub-problem c. The images have to scale up to ensure key points existing.")
    parser.add_argument('folder', help="Image folder.")
    args = parser.parse_args()

    # process
    if args.subproblem == "b":
        print("Sub-problem b:")
        river1 = readRaw(os.path.join(args.folder, RIVER_1_FILENAME), RIVER_HEIGHT, RIVER_WIDTH, RIVER_CHANNEL)
        river2 = readRaw(os.path.join(args.folder, RIVER_2_FILENAME), RIVER_HEIGHT, RIVER_WIDTH, RIVER_CHANNEL)
        SIFT(river1, river2)
    elif args.subproblem == "c":
        print("Sub-problem c:")
        dataImages = [readRaw(os.path.join(args.folder, filename), DIGIT_HEIGHT, DIGIT_WIDTH, DIGIT_CHANNEL) for filename in DIGIT_DATA_FILENAMES]
        sampleImage = readRaw(os.path.join(args.folder, DIGIT_SAMPLE_FILENAME), DIGIT_HEIGHT, DIGIT_WIDTH, DIGIT_CHANNEL)
        nearestImage = bagOfVisualWords(dataImages, sampleImage, args.scale)
        cv2.imshow("Nearest image in database", nearestImage)

    # final
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()