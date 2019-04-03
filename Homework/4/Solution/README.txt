# Informations

+ EE569 Homework #4
+ Date: Mar, 19th, 2019
+ Name: Zongjian Li
+ ID: 6503-3789-43
+ Email: zongjian@usc.edu

# Platforms

Windows 10 (x64)
Matlab R2018b
Visual Studio 2017 & Anaconda3 (python 3.6) & opencv-contrib-python (no higher than 3.4.2.16)

# File Arrangement

Folder "Problem1" is the project folder of Problem 1.
"Problem1.m" is the source code file for Matlab.

Folder "Problem2" is the project folder of Problem 2;
"Problem2.sln" is the solution file for Visual Studio.
"Problem2.py" is the source code file.

# Usage

## Problem1

Before running the script, please modify the image path.

## Problem2

usage: Problem2.py [-h] [--subproblem {b,c}] [--scale f] folder

For USC EE569 2019 spring home work 4 problem 2 by Zongjian Li.

positional arguments:
  folder                Image folder.

optional arguments:
  -h, --help            show this help message and exit
  --subproblem {b,c}, -p {b,c}
                        Choose sub-problem b or c
  --scale f, -s f       Scale factor for sub-problem c. The images have to
                        scale up to ensure key points existing.

Examples:
		Problem2.py -p b ../../HW4_Images
		Problem2.py -p c -s 1.5 ../../HW4_Images