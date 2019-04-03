# Informations

+ EE569 Homework #3
+ Date: Mar, 3rd, 2019
+ Name: Zongjian Li
+ ID: 6503-3789-43
+ Email: zongjian@usc.edu

# Platforms

Windows 10 (x64)
Visual Studio 2017 (C++ latest version)
Matlab R2018b (Only Problem 1 c)

# File Arrangement

Folder "Problem1" is the project folder of Probem1. 
"Problem1.sln" is the solution file for Visual Studio.
"Problem1.cpp" is the source code file.
"Problem1c.m" is the source code file for Matlab.

Folder "Problem2" is the project folder of Probem2;
"Problem2.sln" is the solution file for Visual Studio.
"Problem2.cpp" is the source code file.

# Compile

There is no additional requirement of head files, except standard C/C++ libraries.
Use any C++ compiler (recommended: latest visual c++ compiler) to compile the C++ source codes, or you can open the .sln file and compile with the "Compile" button in GUI.
Please compile under RELEASE mode.

# Run C++ Programs

Compiled program can be run in command line or in the Visual Studio IDE.
If you use Visual Studio to run this program, you should set the input arguments first then click "Run" button.

# Usage

## Problem1

Usage:
        Problem1 [OPTION]... [INPUT FILE / DIR]

Intro:
        Geometric Modification.
        For USC EE569 2019 spring home work 3 problem 1 by Zongjian Li.

Options:
        -p      Sub problems.
                You can choose from "a"(A - geometric transformation), "b"(B - spatial warpping), "c"(C - lens distortion correction).
                NOTE: If you choose problem A, the input should be the folder that contains images, otherwise the input image filename.
                The default is Problem A.
        -s      # of shrinking pixels of the auto detected boundaries of patterns. For problwm A. The default is 1.
        -o      Output filename. The default is "output.raw".
        -h      Height of the input image. The default is 512.
        -w      Width of the input image. The default is 512.
        -c      Number of channels of the input image. The default is 1.

Example:
		./Problem1.exe -p a "../../Home Work No. 3"
		./Problem1.exe -p b "../../Home Work No. 3/hat.raw"
		./Problem1.exe -p c -h 712 -w 1072 "../../Home Work No. 3/classroom.raw"

## Problem1-c

Before running the script, please modify the image path.

## Problem2

Usage:
        Problem2 [OPTION]... [INPUT FILE]

Intro:
        Morphological Processing.
        For USC EE569 2019 spring home work 3 problem 2 by Zongjian Li.

Options:
        -f      Functions.
                You can choose from "s"(Shrink), "t"(Thin), "k"(Skeletonize), "d"(Deer, problem b), "r"(Rice, problem c).
                The default is Shrink.
        -a      Defect area threshold. Connected region with area smaller than this threshold will be consider as defects. Both problem b and c use this parameter. The default is 10.
        -d      Threshold of color distance. Used in problem c to determine whether a pixel is the background. The default is 16.
        -k      Number of types of rice. K in K-means algorithm. Used in problem c. The default is 11.
        -t      Number of trials of K-means. The best result will be taken among trials. Used in problem c. The default is 100.
        -o      Output filename. The default is "output.raw".
        -h      Height of the input image. The default is 375.
        -w      Width of the input image. The default is 375.
        -c      Number of channels of the input image. The default is 1.

Example:
		./Problem2.exe -f t "../../../Home Work No. 3/pattern3.raw"
		./Problem2.exe -f k "../../../Home Work No. 3/pattern4.raw"
		./Problem2.exe -f s -h 691 -w 550 "../../../Home Work No. 3/deer.raw"
		./Problem2.exe -f d -h 691 -w 550 "../../../Home Work No. 3/deer.raw"
		./Problem2.exe -f r -h 500 -w 690 -c 3 "../../../Home Work No. 3/rice.raw"