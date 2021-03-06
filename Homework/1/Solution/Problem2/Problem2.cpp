/************************************
 *  Name: Zongjian Li               *
 *  USC ID: 6503378943              *
 *  USC Email: zongjian@usc.edu     *
 *  Submission Date: 22th,Jan 2019  *
 ************************************/

 /*=================================
 |                                 |
 |              util               |
 |                                 |
 =================================*/

#define  _CRT_SECURE_NO_WARNINGS 

#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

const int UNSIGNED_CHAR_MAX_VALUE = 0xFF;
const int FLOAT_DISPLAY_PRECISION = 5;
const int IMAGE_WIDTH = 256;
const int IMAGE_HEIGHT = 256;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const int GRAY_CHANNELS = 1;
const int COLOR_CHANNELS = 3;
const char* OUTPUT_FILENAME = "output.raw";

/// Mirror reflect padding, use the first/last row/col as axis.
inline int reflectedIndex(const int size, int index) {
	if (index < 0) {
		return -index;
	} else if (index >= size) {
		return 2 * size - index - 2;
	} else {
		return index;
	}
}

inline int flattenedIndex(const int width, const int row, const int col) {
	return width * row + col;
}

template<typename T>
inline float conv(T * data, const int dataHeight, const int dataWidth, const int dataChannels, const int row, const int col, const int ch, float * filter, const int filterHeight, const int filterWidth) {
	int horiShift = filterWidth / 2;
	int vertShift = filterHeight / 2;
	float sum = 0;
	float weightSum = 0;
	for (int i = 0; i < filterHeight; i++) {
		for (int j = 0; j < filterWidth; j++) {
			float w = filter[flattenedIndex(filterWidth, i, j)];
			T d = data[dataChannels * flattenedIndex(dataWidth, reflectedIndex(dataHeight, row + i - vertShift), reflectedIndex(dataWidth, col + j - horiShift)) + ch];
			sum += w * d;
		}
	}
	return sum;
}

inline unsigned char roundAndClipToUnsignedChar(const float value) {
	int rounded = (int)round(value); //I use round() here to get a accurate approximation, while using round() or just casting away decimals visually looks the same.
	if (rounded > UNSIGNED_CHAR_MAX_VALUE) {
		return UNSIGNED_CHAR_MAX_VALUE;
	} else if (rounded < 0x00) {
		return 0x00;
	} else {
		return (unsigned char)rounded;
	}
}

/// Apply given filter to all pixels in given image, without considering relationships between channels.
void applyFilter(unsigned char * input, unsigned char * output, const int dataHeight, const int dataWidth, const int dataChannels, float * filter, const int filterHeight, const int filterWidth) {
	for (int i = 0; i < dataHeight; i++) {
		for (int j = 0; j < dataWidth; j++) {
			for (int k = 0; k < dataChannels; k++) {
				output[dataChannels * flattenedIndex(dataWidth, reflectedIndex(dataHeight, i), reflectedIndex(dataWidth, j)) + k] = roundAndClipToUnsignedChar(conv(input, dataHeight, dataWidth, dataChannels, i, j, k, filter, filterHeight, filterWidth));
			}
		}
	}
}

void normalizeFilter(float * filter, const int size) {
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += filter[i];
	}
	for (int i = 0; i < size; i++) {
		filter[i] /= sum;
	}
}

void printFilter(float * filter, const int height, const int width) {
	cout << setprecision(FLOAT_DISPLAY_PRECISION) << "{" << endl;
	for (int i = 0; i < height; i++) {
		cout << "\t{";
		for (int j = 0; j < width; j++) {
			cout << filter[flattenedIndex(width, i, j)] << ", ";
		}
		cout << "}," << endl;
	}
	cout << endl << "}" << endl << showpoint;
}

/// Calculate PSNR(dB)
float psnr(unsigned char * noiseFree, unsigned char * filtered, const int size) {
	float mse = 0;
	for (int i = 0; i < size; i++) {
		int diff = filtered[i] - noiseFree[i];
		mse += diff * diff;
	}
	mse /= size;
	return 10 * log10(UNSIGNED_CHAR_MAX_VALUE * UNSIGNED_CHAR_MAX_VALUE / mse);
}

void read(const char * filename, void * data, const int size) {
	FILE *file;
	if (!(file = fopen(filename, "rb"))) {
		cout << "Cannot open file: " << filename << endl;
		exit(1);
	}
	fread(data, sizeof(unsigned char), size, file);
	fclose(file);
}

void write(void * data, const int size, const char * filename) {
	FILE *file;
	if (!(file = fopen(filename, "wb"))) {
		cout << "Cannot open file: " << filename << endl;
		exit(1);
	}
	fwrite(data, sizeof(unsigned char), size, file);
	fclose(file);
}

void separate(unsigned char * input, const int size, unsigned char * R, unsigned char * G, unsigned char * B) {
	for (int i = 0; i < size; i++) {
		int baseIndex = COLOR_CHANNELS * i;
		R[i] = input[baseIndex];
		G[i] = input[baseIndex + 1];
		B[i] = input[baseIndex + 2];
	}
}

void combine(unsigned char * R, unsigned char * G, unsigned char * B, unsigned char * output, const int size) {
	for (int i = 0; i < size; i++) {
		int baseIndex = COLOR_CHANNELS * i;
		output[baseIndex] = R[i];
		output[baseIndex + 1] = G[i];
		output[baseIndex + 2] = B[i];
	}
}

/*=================================
|                                 |
|               (a)               |
|                                 |
=================================*/

//--------------(2)----------------

void genUniformFilter(float * filter, const int height, const int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			filter[flattenedIndex(width, i, j)] = 1;
		}
	}
	normalizeFilter(filter, height * width);
}

///Problem 2 - (2) - uniform weight function
void a1(char * noiseFilename, char * noiseFreeFilename, const int filterHeight, const int filterWidth) {

	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFilename, input, IMAGE_SIZE);	

	// Generate filter
	float * filter = new float[filterHeight * filterWidth];
	genUniformFilter(filter, filterHeight, filterWidth);
	cout << "Uniform filter " << filterHeight << "*" << filterWidth << " ";
	printFilter(filter, filterHeight, filterWidth);

	// Apply filter
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH];
	applyFilter((unsigned char *)input, (unsigned char *)output, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, filter, filterHeight, filterWidth);
	delete[] filter;

	// Calculate PSNR(dB)
	unsigned char comp[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFreeFilename, comp, IMAGE_SIZE);
	cout << "Noise image PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)input, IMAGE_SIZE) << showpoint << endl;
	cout << "Filtered image PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)output, IMAGE_SIZE) << showpoint << endl;

	// Write result
	write(output, IMAGE_SIZE * GRAY_CHANNELS, OUTPUT_FILENAME);
}

void genGaussianFilter(float * filter, const int height, const int width, const float sigma) {
	int filterHoriMid = width / 2;
	int filterVertMid = height / 2;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int vDist = i - filterVertMid;
			int hDist = j - filterHoriMid;
			filter[flattenedIndex(width, i, j)] = exp(-(vDist * vDist + hDist * hDist) / (2.0 * pow(sigma, 2))) /* / (sqrt(2.0 * M_PI) * sigma) */; // x * x for integer, pow(x, 2) for float // RHS last part is not necessary, because of the normalize procedure
		}
	}
	normalizeFilter(filter, height * width);
}

///Problem 2 - (2) - gaussian weight function
void a2(char * noiseFilename, char * noiseFreeFilename, const int filterHeight, const int filterWidth, const float filterSigma) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFilename, input, IMAGE_SIZE);

	// Generate filter
	float * filter = new float[filterHeight * filterWidth];
	genGaussianFilter(filter, filterHeight, filterWidth , filterSigma);
	cout << "Gaussian filter " << filterHeight << "*" << filterWidth << " with sigma=" << setprecision(FLOAT_DISPLAY_PRECISION) << filterSigma << showpoint;
	printFilter(filter, filterHeight, filterWidth);

	// Apply filter
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH];
	applyFilter((unsigned char *)input, (unsigned char *)output, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, filter, filterHeight, filterWidth);
	delete[] filter;

	// Calculate PSNR(dB)
	unsigned char comp[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFreeFilename, comp, IMAGE_SIZE);
	cout << "PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)output, IMAGE_SIZE) << showpoint << endl;

	// Write result
	write(output, IMAGE_SIZE * GRAY_CHANNELS, OUTPUT_FILENAME);
}

//--------------(3)----------------

void genBilateralFilterAtPixel(float * filter, const int filterHeight, const int filterWidth, const float sigmaC, const float sigmaS, unsigned char * data, const int dataHeight, const int dataWidth, const int dataChannels, const int row, const int col) {
	int filterHoriMid = filterWidth / 2;
	int filterVertMid = filterHeight / 2;
	for (int i = 0; i < filterHeight; i++) {
		for (int j = 0; j < filterWidth; j++) {
			int vertDist = i - filterVertMid;
			int horiDist = j - filterHoriMid;
			float pixelDistSqr = 0;
			for (int k = 0; k < dataChannels; k++) {
				int chDist = data[dataChannels * (flattenedIndex(dataWidth, reflectedIndex(dataHeight, row - filterVertMid + i), reflectedIndex(dataWidth, col - filterHoriMid + j))) + k] - data[dataChannels * (flattenedIndex(dataWidth, reflectedIndex(dataHeight, row), reflectedIndex(dataWidth, col))) + k];
				pixelDistSqr += chDist * chDist; // A simple distance equation, not suitable for RGB model
			}
			filter[flattenedIndex(filterWidth, i, j)] = exp(-((vertDist * vertDist + horiDist * horiDist) / (2.0 * pow(sigmaC, 2))) - (pixelDistSqr / (2.0 * pow(sigmaS, 2)))); // x * x for integer, pow(x, 2) for float
		}
	}
	normalizeFilter(filter, filterHeight * filterWidth);
}

///Problem 2 - (3)
void a3(char * noiseFilename, char * noiseFreeFilename, const int filterHeight, const int filterWidth, const float filterSigmaC, const float filterSigmaS) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFilename, input, IMAGE_SIZE);

	// Apply bilateral filter
	float * filter = new float[filterHeight * filterWidth];
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			genBilateralFilterAtPixel(filter, filterHeight, filterWidth, filterSigmaC, filterSigmaS, (unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j);
			//printFilter(filter, filterHeight, filterWidth);
			output[i][j] = roundAndClipToUnsignedChar(conv((unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j, 0, filter, filterHeight, filterWidth));
		}
	}
	delete[] filter;

	// Calculate PSNR(dB)
	unsigned char comp[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFreeFilename, comp, IMAGE_SIZE);
	cout << "PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)output, IMAGE_SIZE) << showpoint << endl;

	// Write result
	write(output, IMAGE_SIZE * GRAY_CHANNELS, OUTPUT_FILENAME);
}

//--------------(4)----------------

void genNonlocalMeanFilterAtPixel(float * filter, const int filterHeight, const int filterWidth, const int neighbourHeight, const int neighbourWidth, const float sigma, const float h, unsigned char * data, const int dataHeight, const int dataWidth, const int dataChannels, const int row, const int col) {
	int filterHoriMid = filterWidth / 2;
	int filterVertMid = filterHeight / 2;
	int neighbourHoriMid = neighbourWidth / 2;
	int neighbourVertMid = neighbourHeight / 2;
	for (int i = 0; i < filterHeight; i++) {
		for (int j = 0; j < filterWidth; j++) {
			int winCenterRow = row - filterVertMid + i;
			int winCenterCol = col - filterHoriMid + j;
			long winDistSqr = 0;
			for (int ii = 0; ii < neighbourHeight; ii++) {
				for (int jj = 0; jj < neighbourWidth; jj++) {
					int neighbourRowShift = ii - neighbourVertMid;
					int neighbourColShift = jj - neighbourHoriMid;
					float gaussianFactor = exp(- (neighbourRowShift * neighbourRowShift + neighbourColShift * neighbourColShift) / (2.0 * pow(sigma, 2))) / (sqrt(2.0 * M_PI) * sigma);
					for (int k = 0; k < dataChannels; k++) {
						int chDist = data[dataChannels * flattenedIndex(dataWidth, reflectedIndex(dataHeight, winCenterRow + neighbourRowShift), reflectedIndex(dataWidth, winCenterCol + neighbourColShift)) + k]
							- data[dataChannels * flattenedIndex(dataWidth, reflectedIndex(dataHeight, row + neighbourRowShift), reflectedIndex(dataWidth, col + neighbourColShift)) + k];
						winDistSqr += gaussianFactor * (chDist * chDist);
					}
				}
			}
			filter[flattenedIndex(filterWidth, i, j)] = exp(- winDistSqr / pow(h, 2));
		}
	}
	normalizeFilter(filter, filterHeight * filterWidth);
}

///Problem 2 - (4)
void a4(char * noiseFilename, char * noiseFreeFilename, const int filterHeight, const int filterWidth, const int neighbourHeight, const int neighbourWidth, const float filterSigma, const float filterH) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFilename, input, IMAGE_SIZE);

	// Apply non-local mean filter
	float * filter = new float[filterHeight * filterWidth];
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			genNonlocalMeanFilterAtPixel(filter, filterHeight, filterWidth, neighbourHeight, neighbourWidth, filterSigma, filterH, (unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j);
			output[i][j] = roundAndClipToUnsignedChar(conv((unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j, 0, filter, filterHeight, filterWidth));
		}
	}
	delete[] filter;

	// Calculate PSNR(dB)
	unsigned char comp[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFreeFilename, comp, IMAGE_SIZE);
	cout << "PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)output, IMAGE_SIZE) << showpoint << endl;

	// Write result
	write(output, IMAGE_SIZE * GRAY_CHANNELS, OUTPUT_FILENAME);
}

/*=================================
|                                 |
|               (b)               |
|                                 |
=================================*/

struct elem{
	unsigned char value;
	int row;
	int col;
};

/// Get the median of the given list, not the fastest way, but the easiest way.
elem median(elem * list, const int size) {
	// selection sort (in-place)
	for (int i = 0; i < size - 1; i++) {
		int minIndex = i + 1;
		unsigned char minValue = (list + minIndex)->value;
		for (int j = i + 2; j < size; j++) {
			if ((list + j)->value < minValue) {
				minValue = (list + j)->value;
				minIndex = j;
			}
		}
		elem temp = *(list + i);
		*(list + i) = *(list + minIndex);
		*(list + minIndex) = temp;		
	}
	// get medium
	return *(list + size / 2);
}

void genMedianFilterAtGrayPixel(float * filter, const int filterHeight, const int filterWidth, unsigned char * data, const int dataHeight, const int dataWidth, const int row, const int col) {
	elem * list = new elem[filterHeight * filterWidth];
	int index = 0;
	int filterHoriMid = filterWidth / 2;
	int filterVertMid = filterHeight / 2;
	for (int i = 0; i < filterHeight; i++) {
		for (int j = 0; j < filterWidth; j++) {
			elem * ePtr = list + index;
			ePtr->value = data[flattenedIndex(dataWidth, reflectedIndex(dataHeight, row - filterVertMid + i), reflectedIndex(dataWidth, col - filterHoriMid + j))];
			ePtr->row = i;
			ePtr->col = j;
			index++;
		}
	}
	elem m = median(list, filterHeight * filterWidth);
	delete[] list;
	for (int i = 0; i < filterHeight; i++) {
		for (int j = 0; j < filterWidth; j++) {
			filter[flattenedIndex(filterWidth, i, j)] = i == m.row && j == m.col ? 1 : 0;
		}
	}
	//normalizeFilter(filter, filterHeight * filterWidth);
}

//--------------(1)----------------

void b0(char * noiseFilename, const int filterHeight, const int filterWidth) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Separate
	unsigned char inputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)input, IMAGE_SIZE, (unsigned char *)inputChannels[0], (unsigned char *)inputChannels[1], (unsigned char *)inputChannels[2]);

	// Median filter (channels separated)
	float * filter = new float[filterHeight * filterWidth];
	unsigned char outputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
		for (int i = 0; i < IMAGE_HEIGHT; i++) {
			for (int j = 0; j < IMAGE_WIDTH; j++) {
				genMedianFilterAtGrayPixel(filter, filterHeight, filterWidth, (unsigned char *)inputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, i, j);
				outputChannels[ch][i][j] = roundAndClipToUnsignedChar(conv((unsigned char *)inputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j, 0, filter, filterHeight, filterWidth));
			}
		}
	}	
	delete[] filter;

	// Merge
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	combine((unsigned char *)outputChannels[0], (unsigned char *)outputChannels[1], (unsigned char *)outputChannels[2], (unsigned char *)output, IMAGE_SIZE);

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

void b1(char * noiseFilename, const int filterHeight, const int filterWidth) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Separate
	unsigned char inputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)input, IMAGE_SIZE, (unsigned char *)inputChannels[0], (unsigned char *)inputChannels[1], (unsigned char *)inputChannels[2]);

	// Uniformfilter filter (channels separated)
	float * filter = new float[filterHeight * filterWidth];
	genUniformFilter(filter, filterHeight, filterWidth);
	unsigned char outputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
		applyFilter((unsigned char *)inputChannels[ch], (unsigned char *)outputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, (float *)filter, filterHeight, filterWidth);
	}
	delete[] filter;

	// Merge
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	combine((unsigned char *)outputChannels[0], (unsigned char *)outputChannels[1], (unsigned char *)outputChannels[2], (unsigned char *)output, IMAGE_SIZE);

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

void b2(char * noiseFilename, const int filterHeight, const int filterWidth, const float filterSigma) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Separate
	unsigned char inputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)input, IMAGE_SIZE, (unsigned char *)inputChannels[0], (unsigned char *)inputChannels[1], (unsigned char *)inputChannels[2]);

	// Gaussian filter (channels separated)
	float * filter = new float[filterHeight * filterWidth];
	genGaussianFilter(filter, filterHeight, filterWidth, filterSigma);
	unsigned char outputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
		applyFilter((unsigned char *)inputChannels[ch], (unsigned char *)outputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, (float *)filter, filterHeight, filterWidth);
	}
	delete[] filter;

	// Merge
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	combine((unsigned char *)outputChannels[0], (unsigned char *)outputChannels[1], (unsigned char *)outputChannels[2], (unsigned char *)output, IMAGE_SIZE);

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

void b3a(char * noiseFilename, const int filterHeight, const int filterWidth, const float filterSigmaC, const float filterSigmaS) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Separate
	unsigned char inputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)input, IMAGE_SIZE, (unsigned char *)inputChannels[0], (unsigned char *)inputChannels[1], (unsigned char *)inputChannels[2]);

	// Bilateral filter (channels separated)
	float * filter = new float[filterHeight * filterWidth];
	unsigned char outputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
		for (int i = 0; i < IMAGE_HEIGHT; i++) {
			for (int j = 0; j < IMAGE_WIDTH; j++) {
				genBilateralFilterAtPixel(filter, filterHeight, filterWidth, filterSigmaC, filterSigmaS, (unsigned char *)inputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j);
				outputChannels[ch][i][j] = roundAndClipToUnsignedChar(conv((unsigned char *)inputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j, 0, filter, filterHeight, filterWidth));
			}
		}
	}
	delete[] filter;

	// Merge
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	combine((unsigned char *)outputChannels[0], (unsigned char *)outputChannels[1], (unsigned char *)outputChannels[2], (unsigned char *)output, IMAGE_SIZE);

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

void b3b(char * noiseFilename, const int filterHeight, const int filterWidth, const float filterSigmaC, const float filterSigmaS) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Bilateral filter (channels non-separated)
	float * filter = new float[filterHeight * filterWidth];
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			genBilateralFilterAtPixel(filter, filterHeight, filterWidth, filterSigmaC, filterSigmaS, (unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS, i, j); // the filter is gen by using all RGB informations in the image
			for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
				output[i][j][ch] = roundAndClipToUnsignedChar(conv((unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS, i, j, ch, filter, filterHeight, filterWidth));
			}			
		}
	}
	delete[] filter;

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

void b4a(char * noiseFilename, const int filterHeight, const int filterWidth, const int neighbourHeight, const int neighbourWidth, const float filterSigma, const float filterH) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Separate
	unsigned char inputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)input, IMAGE_SIZE, (unsigned char *)inputChannels[0], (unsigned char *)inputChannels[1], (unsigned char *)inputChannels[2]);

	// Non-local mean filter (channels separated)
	float * filter = new float[filterHeight * filterWidth];
	unsigned char outputChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
		for (int i = 0; i < IMAGE_HEIGHT; i++) {
			for (int j = 0; j < IMAGE_WIDTH; j++) {
				genNonlocalMeanFilterAtPixel(filter, filterHeight, filterWidth, neighbourHeight, neighbourWidth, filterSigma, filterH, (unsigned char *)inputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j);
				outputChannels[ch][i][j] = roundAndClipToUnsignedChar(conv((unsigned char *)inputChannels[ch], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j, 0, filter, filterHeight, filterWidth));
			}
		}
	}
	delete[] filter;

	// Merge
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	combine((unsigned char *)outputChannels[0], (unsigned char *)outputChannels[1], (unsigned char *)outputChannels[2], (unsigned char *)output, IMAGE_SIZE);

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

void b4b(char * noiseFilename, const int filterHeight, const int filterWidth, const int neighbourHeight, const int neighbourWidth, const float filterSigma, const float filterH) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFilename, input, IMAGE_SIZE * COLOR_CHANNELS);

	// Non-local mean filter (channels non-separated)
	float * filter = new float[filterHeight * filterWidth];
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			genNonlocalMeanFilterAtPixel(filter, filterHeight, filterWidth, neighbourHeight, neighbourWidth, filterSigma, filterH, (unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS, i, j); // the filter is gen by using all RGB informations in the image
			for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
				output[i][j][ch] = roundAndClipToUnsignedChar(conv((unsigned char *)input, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS, i, j, ch, filter, filterHeight, filterWidth));
			}
		}
	}
	delete[] filter;

	// Write result
	write(output, IMAGE_SIZE * COLOR_CHANNELS, OUTPUT_FILENAME);
}

/*=================================
|                                 |
|               (c)               |
|                                 |
=================================*/

void expand(unsigned char * input, float * output, const int size) {
	for (int i = 0; i < size; i++) {
		output[i] = input[i];
	}
}

void forwardTransfer(float * data, const int size) {
	for (int i = 0; i < size; i++) {
		data[i] = 2.0 * sqrt(data[i] + 0.375);
	}
}

void inverseTransfer(float * data, const int size, bool biased) {
	float additional = biased ? -0.25 : 0;
	for (int i = 0; i < size; i++) {
		data[i] = pow(data[i] / 2.0, 2) - 0.125 + additional;
	}
}

void contract(float * input, unsigned char * output, const int size) {
	for (int i = 0; i < size; i++) {
		output[i] = roundAndClipToUnsignedChar(input[i]);
	}
}

//------------(1 & 2)--------------

/// Gaussian low pass filter, BM3D will be used in MATLAB
void c1(char * noiseFilename, char * noiseFreeFilename, const int filterHeight, const int filterWidth, const float filterSigma, const bool biased) {
	// Read image
	unsigned char input[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFilename, input, IMAGE_SIZE);

	// Forward Transfer
	float transInput[IMAGE_HEIGHT][IMAGE_WIDTH];
	expand((unsigned char *)input, (float *)transInput, IMAGE_SIZE);
	forwardTransfer((float *)transInput, IMAGE_SIZE);

	// Generate filter
	float * filter = new float[filterHeight * filterWidth];
	genGaussianFilter(filter, filterHeight, filterWidth, filterSigma);
	cout << "Gaussian filter " << filterHeight << "*" << filterWidth << " with sigma=" << setprecision(FLOAT_DISPLAY_PRECISION) << filterSigma << showpoint;
	printFilter(filter, filterHeight, filterWidth);	

	// Apply filter
	float transOutput[IMAGE_HEIGHT][IMAGE_WIDTH];
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			transOutput[i][j] = conv((float *) transInput, IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNELS, i, j, 0, filter, filterHeight, filterWidth);
		}
	}
	delete[] filter;

	// Inverse Transfer
	inverseTransfer((float *)transOutput, IMAGE_SIZE, biased);
	unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH];
	contract((float *)transOutput, (unsigned char *)output, IMAGE_SIZE);

	// Calculate PSNR(dB)
	unsigned char comp[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFreeFilename, comp, IMAGE_SIZE);
	cout << "Noise image PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)input, IMAGE_SIZE) << showpoint << endl;
	cout << "Filtered image PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)comp, (unsigned char *)output, IMAGE_SIZE) << showpoint << endl;

	// Write result
	write(output, IMAGE_SIZE * GRAY_CHANNELS, OUTPUT_FILENAME);
}

/*=================================
|                                 |
|              main               |
|                                 |
=================================*/
const char * PROG_NAME = "Problem2";
const char * COMMAND_A1 = "a1";
const char * COMMAND_A2 = "a2";
const char * COMMAND_A3 = "a3";
const char * COMMAND_A4 = "a4";
const char * COMMAND_B0 = "b0";
const char * COMMAND_B1 = "b1";
const char * COMMAND_B2 = "b2";
const char * COMMAND_B3_A = "b3a";
const char * COMMAND_B3_B = "b3b";
const char * COMMAND_B4_A = "b4a";
const char * COMMAND_B4_B = "b4b";
const char * COMMAND_C1 = "c1";
const char * COMMAND_COLOR = "color";
const char * COMMAND_GRAY = "gray";
const char * OPTION_HELP = "-h";
const char * ERR_COMMAND = "Wrong command!";
const char * ERR_ARGUMENTS = "Wrong argument(s)!";

void printMainMannual() {
	cout << "Usage: " << PROG_NAME << " <command> " << endl;
	cout << endl;
	cout << "where <command> is one of:" << endl;
	cout << "\t" << COMMAND_A1 << ", " << COMMAND_A2 << ", " << COMMAND_A3 << ", " << COMMAND_A4 << ", " << endl;
	cout << "\t" << COMMAND_B0 << ", " << COMMAND_B1 << ", " << COMMAND_B2 << ", " << COMMAND_B3_A << ", " << COMMAND_B3_B << ", " << COMMAND_B4_A << ", " << COMMAND_B4_B << ", " << endl;
	cout << "\t" << COMMAND_C1 << ", " <<  endl;
	cout << "\t" << COMMAND_COLOR << ", " << COMMAND_GRAY << endl;
	cout << endl;
	cout << PROG_NAME << " <command> " << OPTION_HELP << "\t" << "quick help on <command>" << endl;
	cout << endl;
	cout << "You do not need to input the size of the input images, because in this problem input images are static." << endl;
	cout << "PSNR(dB) will be print into standard output if available." << endl;
	cout << "An output raw image file \"" << OUTPUT_FILENAME << "\" will be create at the execution directory." << endl;
}

void printA1Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 a) (2) uniform: using uniform weight function low pass filter to remove added noise in pepper image." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_A1 << " <noise filename> <noise-free filename> <kernel height: odd positive integer> <kernel width: odd positive integer>" << endl;
}

void printA2Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 a) (2) gaussian: using gaussian weight function low pass filter to remove added noise in pepper image." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_A2 << " <noise filename> <noise-free filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma: positive decimal>" << endl;
}

void printA3Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 a) (3): using bilateral filter to remove added noise in pepper image." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_A3 << " <noise filename> <noise-free filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma c: positive decimal> <sigma s: positive decimal>" << endl;
}

void printA4Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 a) (4): using non-local mean filter to remove added noise in pepper image." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_A4 << " <noise filename> <noise-free filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <neighbour height: odd positive integer> <neighbour width: odd positive integer> <sigma: positive decimal> <h: positive decimal>" << endl;
}

void printB0Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using median filter (separate  channels) to remove impulse noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B0 << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma: positive decimal> <biased: bool 0 or 1>" << endl;
}

void printB1Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using uniform weight function low pass filter (separate channels) to remove other noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B1 << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer>" << endl;
}

void printB2Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using gaussian weight function low pass filter (separate channels) to remove other noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B2 << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma: positive decimal>" << endl;
}

void printB3AMannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using bilateral filter (separate channels) to remove other noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B3_A << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma c: positive decimal> <sigma s: positive decimal>" << endl;
}

void printB3BMannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using bilateral filter (combined channels) to remove other noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B3_B << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma c: positive decimal> <sigma s: positive decimal>" << endl;
}

void printB4AMannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using non-local mean filter (separate channels) to remove other noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B4_A << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <neighbour height: odd positive integer> <neighbour width: odd positive integer> <sigma: positive decimal> <h: positive decimal>" << endl;
}

void printB4BMannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 b): using non-local mean filter (combined channels) to remove other noise in rose_color_noise.raw or other color denoising results." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_B4_B << " <noise filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <neighbour height: odd positive integer> <neighbour width: odd positive integer> <sigma: positive decimal> <h: positive decimal>" << endl;
}

void printC1Mannual() {
	cout << endl;
	cout << "This is a program for Homework1 Problem2 c) gaussian: using gaussian weight function low pass filter to remove short noise in pepper_dark_noise image." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_C1 << " <noise filename> <noise-free filename> <kernel height: odd positive integer> <kernel width: odd positive integer> <sigma: positive decimal> <biased: bool 0 or 1>" << endl;
}

void printColorMannual() {
	cout << endl;
	cout << "Calculate PSNR and differences (color image)." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_COLOR << " <filtered filename> <noise-free filename>" << endl;
}

void printGrayMannual() {
	cout << endl;
	cout << "Calculate PSNR and differences (gray image)." << endl;
	cout << endl;
	cout << "Usage: " << PROG_NAME << " " << COMMAND_GRAY << " <noise filename> <noise-free filename>" << endl;
}

///helper function
void calcColor(char * filteredFilename, char * noiseFreeFilename) { // for color image
	// PSNR
	unsigned char filtered[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(filteredFilename, filtered, IMAGE_SIZE * COLOR_CHANNELS);
	unsigned char noiseFree[IMAGE_HEIGHT][IMAGE_WIDTH][COLOR_CHANNELS];
	read(noiseFreeFilename, noiseFree, IMAGE_SIZE * COLOR_CHANNELS);	
	cout << "PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)noiseFree, (unsigned char *)filtered, IMAGE_SIZE* COLOR_CHANNELS) << showpoint << endl;

	// Difference
	unsigned char filteredChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)filtered, IMAGE_SIZE, (unsigned char *)filteredChannels[0], (unsigned char *)filteredChannels[1], (unsigned char *)filteredChannels[2]);
	unsigned char noiseFreeChannels[COLOR_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
	separate((unsigned char *)noiseFree, IMAGE_SIZE, (unsigned char *)noiseFreeChannels[0], (unsigned char *)noiseFreeChannels[1], (unsigned char *)noiseFreeChannels[2]);
	int hist[UNSIGNED_CHAR_MAX_VALUE * 2 + 1];
	for (int ch = 0; ch < COLOR_CHANNELS; ch++) {		
		memset(hist, 0, sizeof(int) * (UNSIGNED_CHAR_MAX_VALUE * 2 + 1));
		for (int i = 0; i < IMAGE_HEIGHT; i++) {
			for (int j = 0; j < IMAGE_WIDTH; j++) {
				hist[filteredChannels[ch][i][j] - noiseFreeChannels[ch][i][j] + UNSIGNED_CHAR_MAX_VALUE]++;
			}
		}
		// Print
		cout << "Channel " << ch << "{";
		for (int i = -UNSIGNED_CHAR_MAX_VALUE; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
			cout << "{" << i << ", " << hist[i + UNSIGNED_CHAR_MAX_VALUE] << "}, ";
		}
		cout << "}" << endl;
	}
}

///helper function
void calcGray(char * filteredFilename, char * noiseFreeFilename) { // gray image
	// Difference histogram
	unsigned char noise[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(filteredFilename, noise, IMAGE_SIZE * GRAY_CHANNELS);
	unsigned char noiseFree[IMAGE_HEIGHT][IMAGE_WIDTH];
	read(noiseFreeFilename, noiseFree, IMAGE_SIZE * GRAY_CHANNELS);
	cout << "PSNR(dB)=" << setprecision(FLOAT_DISPLAY_PRECISION) << psnr((unsigned char *)noiseFree, (unsigned char *)noise, IMAGE_SIZE* GRAY_CHANNELS) << showpoint << endl;

	// Difference histogram
	int hist[UNSIGNED_CHAR_MAX_VALUE * 2 + 1];
	memset(hist, 0, sizeof(int) * (UNSIGNED_CHAR_MAX_VALUE * 2 + 1));
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			hist[noise[i][j] - noiseFree[i][j] + UNSIGNED_CHAR_MAX_VALUE]++;
		}
	}
	// Print
	cout << "{";
	for (int i = -UNSIGNED_CHAR_MAX_VALUE; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
		cout << "{" << i << ", " << hist[i + UNSIGNED_CHAR_MAX_VALUE] << "}, ";
	}
	cout << "}" << endl;
}

int main(int argc, char *argv[]) {
	// check n: 1. n is odd, 2. n less than 256
	if (argc == 1) {
		printMainMannual();
	} else if (strcmp(argv[1], COMMAND_A1) == 0) {					//a1
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printA1Mannual();
		} else if (argc == 6) {
			int h = atoi(argv[4]);
			int w = atoi(argv[5]);
			if (h % 2 == 1 && w % 2 == 1) {
				a1(argv[2], argv[3], h, w);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printA1Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printA1Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_A2) == 0) {					//a2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printA2Mannual();
		} else if (argc == 7) {
			int h = atoi(argv[4]);
			int w = atoi(argv[5]);
			float s = atof(argv[6]);
			if (h % 2 == 1 && w % 2 == 1 && s > 0) {
				a2(argv[2], argv[3], h, w, s);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printA2Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printA2Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_A3) == 0) {					//a3
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printA3Mannual();
		} else if (argc == 8) {
			int h = atoi(argv[4]);
			int w = atoi(argv[5]);
			float sc = atof(argv[6]);
			float ss = atof(argv[7]);
			if (h % 2 == 1 && w % 2 == 1 && sc > 0 && ss > 0) {
				a3(argv[2], argv[3], h, w, sc, ss);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printA3Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printA3Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_A4) == 0) {					//a4
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printA4Mannual();
		} else if (argc == 10) {
			int wh = atoi(argv[4]);
			int ww = atoi(argv[5]);
			int nh = atoi(argv[6]);
			int nw = atoi(argv[7]);
			float s = atof(argv[8]);
			float h = atof(argv[9]);
			if (wh % 2 == 1 && ww % 2 == 1 && nh % 2 == 1 && nw % 2 == 1 && s > 0 && h > 0) {
				a4(argv[2], argv[3], wh, ww, nh, nw, s, h);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printA4Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printA4Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_C1) == 0) {					//c1
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printC1Mannual();
		} else if (argc == 8) {
			int h = atoi(argv[4]);
			int w = atoi(argv[5]);
			float s = atof(argv[6]);
			bool b = atoi(argv[7]);
			if (h % 2 == 1 && w % 2 == 1 && s > 0) {
				c1(argv[2], argv[3], h, w, s, b);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printC1Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printC1Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_B0) == 0) {					//b0
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB0Mannual();
		} else if (argc == 5) {
			int h = atoi(argv[3]);
			int w = atoi(argv[4]);
			if (h % 2 == 1 && w % 2 == 1) {
				b0(argv[2], h, w);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB0Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB0Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_B1) == 0) {					//b1
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB1Mannual();
		} else if (argc == 5) {
			int h = atoi(argv[3]);
			int w = atoi(argv[4]);
			if (h % 2 == 1 && w % 2 == 1) {
				b1(argv[2], h, w);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB1Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB1Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_B2) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB2Mannual();
		} else if (argc == 6) {
			int h = atoi(argv[3]);
			int w = atoi(argv[4]);
			float s = atof(argv[5]);
			if (h % 2 == 1 && w % 2 == 1 && s > 0) {
				b2(argv[2], h, w, s);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB2Mannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB2Mannual();
		}
	} else if (strcmp(argv[1], COMMAND_B3_A) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB3AMannual();
		} else if (argc == 7) {
			int h = atoi(argv[3]);
			int w = atoi(argv[4]);
			float sc = atof(argv[5]);
			float ss = atof(argv[6]);
			if (h % 2 == 1 && w % 2 == 1 && sc > 0 && ss > 0) {
				b3a(argv[2], h, w, sc, ss);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB3AMannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB3AMannual();
		}
	} else if (strcmp(argv[1], COMMAND_B3_B) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB3BMannual();
		} else if (argc == 7) {
			int h = atoi(argv[3]);
			int w = atoi(argv[4]);
			float sc = atof(argv[5]);
			float ss = atof(argv[6]);
			if (h % 2 == 1 && w % 2 == 1 && sc > 0 && ss > 0) {
				b3b(argv[2], h, w, sc, ss);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB3BMannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB3BMannual();
		}
	} else if (strcmp(argv[1], COMMAND_B4_A) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB4AMannual();
		} else if (argc == 9) {
			int wh = atoi(argv[3]);
			int ww = atoi(argv[4]);
			int nh = atoi(argv[5]);
			int nw = atoi(argv[6]);
			float s = atof(argv[7]);
			float h = atof(argv[8]);
			if (wh % 2 == 1 && ww % 2 == 1 && nh % 2 == 1 && nw % 2 == 1 && s > 0 && h > 0) {
				b4a(argv[2], wh, ww, nh, nw, s, h);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB4AMannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB4AMannual();
		}
	} else if (strcmp(argv[1], COMMAND_B4_B) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printB4BMannual();
		} else if (argc == 9) {
			int wh = atoi(argv[3]);
			int ww = atoi(argv[4]);
			int nh = atoi(argv[5]);
			int nw = atoi(argv[6]);
			float s = atof(argv[7]);
			float h = atof(argv[8]);
			if (wh % 2 == 1 && ww % 2 == 1 && nh % 2 == 1 && nw % 2 == 1 && s > 0 && h > 0) {
				b4b(argv[2], wh, ww, nh, nw, s, h);
			} else {
				cout << ERR_ARGUMENTS << endl << endl;
				printB4BMannual();
			}
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printB4BMannual();
		}
	} else if (strcmp(argv[1], COMMAND_COLOR) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printColorMannual();
		} else if (argc == 4) {
			calcColor(argv[2], argv[3]);
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printColorMannual();
		}
	} else if (strcmp(argv[1], COMMAND_GRAY) == 0) {					//b2
		if (argc == 3 && strcmp(argv[2], OPTION_HELP) == 0) {
			printGrayMannual();
		} else if (argc == 4) {
			calcGray(argv[2], argv[3]);
		} else {
			cout << ERR_ARGUMENTS << endl << endl;
			printGrayMannual();
		}
	} else {
		cout << ERR_COMMAND << endl << endl;
		printMainMannual();
	}
	return 0;
}