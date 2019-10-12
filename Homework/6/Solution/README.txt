# Informations

+ EE569 Homework #6
+ Date: Apr, 28th, 2019
+ Name: Zongjian Li
+ ID: 6503-3789-43
+ Email: zongjian@usc.edu

# Platforms

Windows 10 (x64)
Visual Studio 2017 & Anaconda3 (python 3.6)

# File Arrangement

Folder "Problem1" is the project folder of Problem 1;
"saab_compact.py" is a modified version of code in https://github.com/davidsonic/Interpretable_CNN.
"data.py" is for loading MNIST data, same as that in https://github.com/davidsonic/Interpretable_CNN.
"laws.py" laws filtering model.
"p2-ver-wrap.py" is the source code file for sub problem 2 (calls saab_compact.py).
"p2-ver-mine.py" is the source code file for sub problem 2 (rewrite by myselt).
"p3-train.py" for training FF-CNN and save parameters.
"p3-test.py" for testing FF-CNN with parameters saved by "p3-train.py", and save FF-CNN's output.
"p3-ensemble.py" train the ensemble system using output features saved by "p3-test.py" and test its accuracy.

# Usage

## Problem1

Please add argument -h to see the usage of each script.