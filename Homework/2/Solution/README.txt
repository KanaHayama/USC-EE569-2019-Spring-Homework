# Informations

+ EE569 Homework #2
+ Date: Feb, 12th, 2019
+ Name: Zongjian Li
+ ID: 6503-3789-43
+ Email: zongjian@usc.edu

# Platforms

Windows 10 (x64)
Visual Studio 2017 (C++ latest version)
Matlab R2018b (Only BM3D sub-problem)

# File Arrangement

Folder "Problem1" is the project folder of Probem1. 
"main.m" is the source code file for Matlab.

Folder "Problem2" is the project folder of Probem2;
"Problem2.sln" is the solution file for Visual Studio.
"Problem2.cpp" is the source code file.

# Compile

There is no additional requirement of head files, except standard C/C++ libraries.
Use any C++ compiler (recommended: latest visual c++ compiler) to compile the C++ source codes, or you can open the .sln file and compile with the "Compile" button in GUI.

# Run C++ Programs

Compiled program can be run in command line or in the Visual Studio IDE.
If you use Visual Studio to run this program, you should set the input arguments first then click "Run" button.

# Usage

## Problem1

	1. Install the toolboxs. Let the "edges" and "toolbox" folders be in the same folder with main.m. (Using pre-trained model, BSDS500 data set not needed)
	2. Modify the arguments in "setEnvironment" function, such as path of the pig.raw, tiger.raw and others.
	3. Run the main.m, wait, then you will get result .mat files containing all 3 detectors' measurements for both images in pwd.

## Problem2

Usage:
        Problem2 [OPTION]... [INPUT FILE]

Intro:
        Digital image half-toning.
        For USC EE569 2019 spring home work 2 problem 2 by Zongjian Li.

Options:
        -a      Half-toning method.
                You can choose from "ft"(Fix Thresholding), "rt"(Random Thresholding), "d"(Dithering), "ed"(Error Diffusion), "sed"(Separable Error Diffusion), "mbvq"(MBVQ-based Error Diffusion).
                The default is Random Thresholding.
        -s      Random seed for Random Thresholding method. The default is 0.
        -d      Height / Width of dithering matrix used in Dithering method. Must be a power of 2, smaller than input image size. The default is 32.
        -m      Error diffusion matrix.
                You can choose from "fs"(Floyd-Steinberg's), "jjn"(Jarvis, Judice, and Ninke (JJN)), "s"(Stucki).
                The default is Floyd-Steinberg's.
        -so     Scanning Order.
                You can choose from "r"(Raster), "s"(Serpentine).
                The default is Serpentine.
        -o      Output filename. The default is "output.raw".
        -h      Height of the input image. The default is 400.
        -w      Width of the input image. The default is 600.
        -c      Number of channels of the input image. The default is 1.

Examples:
	-a ft ../../HW2_images/bridge.raw
	-a d -d 2 ../../HW2_images/bridge.raw
	-a d -d 8 ../../HW2_images/bridge.raw
	-a d -d 32 ../../HW2_images/bridge.raw
	-a ed ../../HW2_images/bridge.raw
	-a ed -so r ../../HW2_images/bridge.raw
	-a ed -m jjn ../../HW2_images/bridge.raw
	-a ed -m s ../../HW2_images/bridge.raw
	-a ed -m s -so r ../../HW2_images/bridge.raw
	-a sed -w 500 -h 375 -c 3 ../../HW2_images/bird.raw
	-a mbvq -w 500 -h 375 -c 3 ../../HW2_images/bird.raw
	-a mbvq -so r -w 500 -h 375 -c 3 ../../HW2_images/bird.raw
	-a mbvq -m jjn -w 500 -h 375 -c 3 ../../HW2_images/bird.raw
	-a mbvq -m jjn -so r -w 500 -h 375 -c 3 ../../HW2_images/bird.raw
	-a mbvq -m s -w 500 -h 375 -c 3 ../../HW2_images/bird.raw
	-a mbvq -m s -so r -w 500 -h 375 -c 3 ../../HW2_images/bird.raw