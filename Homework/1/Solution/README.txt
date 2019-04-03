# Informations

+ EE569 Homework #1
+ Date: Jan, 22nd, 2019
+ Name: Zongjian Li
+ ID: 6503-3789-43
+ Email: zongjian@usc.edu

# Platforms

Windows 10 (x64)
Visual Studio 2017 (C++ latest version)
Matlab R2017b (Only BM3D sub-problem)

# File Arrangement

Folder "Problem1" is the project folder of Probem1. 
"Problem1.sln" is the solution file for Visual Studio.
"Problem1.cpp" is the source code file.

Folder "Problem2" is the project folder of Probem2 (except Problem2-{c}-BM3D);
"Problem2.sln" is the solution file for Visual Studio.
"Problem2.cpp" is the source code file.

Folder "BM3D" contains Matlab .m file for Problem2-{c}-BM3D
"main.m" is the source code file to solve the problem, it is wrote by me and it invokes the BM3D function.
other files are downloaded from http://www.cs.tut.fi/~foi/GCF-BM3D/. *I leave them into the zip file according to the requirment of this sub problem, you can replace them if you want.*

# Compile

There is no additional requirement of head files, except standard C/C++ libraries.
Use any C++ compiler (recommended: latest visual c++ compiler) to compile the C++ source codes, or you can open the .sln file and compile with the "Compile" button in GUI.

# Run C++ Programs

Compiled program can be run in command line or in the Visual Studio IDE.
If you use Visual Studio to run this program, you should set the input arguments first then click "Run" button.

# Usage

## Problem1

Problem1.exe <a|b|ca|cb> <input raw filename>
Where a, b, ca, cb (case sensitive) are the sub problems in Problem 1.

Command "a" is for bilinear demosaicing.
Command "b" is for MHC demosaicing.
Command "ca" is the method A of histogram manipulation.
Command "cb" is the method B of histogram manipulation.

You do not need to input the size of input images, because in this problem input images are static.
Data of histogram will be printed into standard output (if available).
An output raw image file "output.raw" will be create at the execution directory.

Examples:
+ Bileaner demosaicing: Problem1.exe a cat.raw
+ MHC demosaicing: Problem1.exe b cat.raw
+ Histogram manipulation method A: Problem1.exe ca rose_mix.raw
+ Histogram manipulation method B: Problem1.exe cb rose_mix.raw

## Problem2

Problem2 <command> ...

where <command> is one of:
	a1, a2, a3, a4,
	b0, b1, b2, b3a, b3b, b4a, b4b,
	c1,
	color, gray

Problem2 <command> -h   quick help on <command>

Command "a1" is for uniform filter (gray image). The arguments are <noise image> <noise-free image> <filter height> <filter width>.
Command "a2" is for gaussian filter (gray image). The arguments are <noise image> <noise-free image> <filter height> <filter width> <sigma>.
Command "a3" is for bilateral filter (gray image). The arguments are <noise image> <noise-free image> <filter height> <filter width> <sigma c> <sigma s>.
Command "a4" is for non-local mean filter (gray image). The arguments are <noise image> <noise-free image> <filter height> <filter width> <neighbour height> <neighbour width> <sigma> <h>.
Command "b0" is for median filter (color image). The arguments are <noise image> <noise-free image> <filter height> <filter width>.
Command "b1" is for uniform filter (color image). The arguments are <noise image> <noise-free image> <filter height> <filter width>.
Command "b2" is for gaussian filter (color image). The arguments are <noise image> <noise-free image> <filter height> <filter width> <sigma>.
Command "b3a" is for bilateral filter (color image, perform on RGB separately). The arguments are <noise image> <noise-free image> <filter height> <filter width> <sigma c> <sigma s>.
Command "b3b" is for bilateral filter (color image). The arguments are <noise image> <noise-free image> <filter height> <filter width> <sigma c> <sigma s>.
Command "b4a" is for non-local mean filter (color image, perform on RGB separately). The arguments are <noise image> <noise-free image> <filter height> <filter width> <neighbour height> <neighbour width> <sigma> <h>.
Command "b4b" is for non-local mean filter (color image). The arguments are <noise image> <noise-free image> <filter height> <filter width> <neighbour height> <neighbour width> <sigma> <h>.
Command "c1" is for short noise denoiseing using gaussian filter.  The arguments are <noise image> <noise-free image> <filter height> <filter width> <sigma> <biased flag (0 or 1)>. Short noise denoising using BM3D is not implemented here.
Command "color" and "gray" is for calculating PSNR overall and the difference in each channel, only for debugging.

Heights and widths of filter arguments should be odd integers and should be smaller than the height or width of input images.
You do not need to input the size of the input images, because in this problem input images are static.
PSNR(dB) will be print into standard output (if available).
An output raw image file "output.raw" will be create at the execution directory.
If you want to cascade filters, you need to run this program multiple times.

Examples:
+ uniform filter (gray): Problem2.exe a1 pepper_uni.raw pepper.raw 5 5
+ gaussian filter (gray): Problem2.exe a2 pepper_uni.raw pepper.raw 5 5 2.0
+ bilateral filter (gray): Problem2.exe a3 pepper_uni.raw pepper.raw 5 5 2.0 64.0
+ non-local mean filter (gray): Problem2.exe a4 pepper_uni.raw pepper.raw 11 11 5 5 2.0 64.0
+ median filter (color): Problem2 b0 rose_color_noise.raw rose_color.raw 3 3
+ uniform filter (color): Problem2.exe b1 rose_color_noise.raw rose_color.raw 5 5
+ gaussian filter (color): Problem2.exe b2 rose_color_noise.raw rose_color.raw 5 5 2.0
+ bilateral filter (color, separate): Problem2.exe b3a rose_color_noise.raw rose_color.raw 5 5 2.0 64.0
+ bilateral filter (color): Problem2.exe b3b rose_color_noise.raw rose_color.raw 5 5 2.0 64.0
+ non-local mean filter (color, separate): Problem2.exe b4a rose_color_noise.raw rose_color.raw 11 11 5 5 2.0 64.0
+ non-local mean filter (color): Problem2.exe b4b rose_color_noise.raw rose_color.raw 11 11 5 5 2.0 64.0
+ short noise denoising (gray, gaussian): Problem2.exe c1 pepper_dark_noise.raw pepper_dark.raw 5 5 2.0 1

## Problem2 - BM3D

Make sure you installed BM3D function into the directory of "BM3D", then open "main.m".
You need to modify the filepath (directory) to indicate where to find input images. Name of inpur images will be "pepper_dark_noise.raw" and "pepper_dark.raw", you may not modify these.
You can change the sigma if you want, the default value I used is 1, that is because the after the transformation is performed the the noise will be gaussian noise with unit variation.
Then click the "Run" button, then you will get three images. The first one is noise image, the second one is biased denoised image, the third one is unbiased denoised image, PSNR will be printed in the output.

# Review Source Code

I enrolled in this course a few days ago, I do not have enough time to re-arrange my code framework. It may be a little hard to read.
