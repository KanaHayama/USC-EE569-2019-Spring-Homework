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
#include <math.h>

using namespace std;

const char* SUB_A_COMMAND = "a";
const char* SUB_B_COMMAND = "b";
const char* SUB_C_A_COMMAND = "ca";
const char* SUB_C_B_COMMAND = "cb";

const char* OUTPUT_FILENAME = "output.raw";

int printUsage() {
	cout << "Usage:" << endl;
	cout << "\t" << "Problem1 <" << SUB_A_COMMAND << "|" << SUB_B_COMMAND << "|" << SUB_C_A_COMMAND << "|" << SUB_C_B_COMMAND << "> <input raw filename>" << endl; //TODO: multiple files
	cout << "\t\t" << "Where " << SUB_A_COMMAND << ", " << SUB_B_COMMAND << ", " << SUB_C_A_COMMAND << ", " << SUB_C_B_COMMAND << " (case sensitive) are the sub problems in Problem 1." << endl;
	cout << "\t\t" << SUB_A_COMMAND << " is bilinear demosaicing;" << endl;
	cout << "\t\t" << SUB_B_COMMAND << " is MHC demosaicing;" << endl;
	cout << "\t\t" << SUB_C_A_COMMAND << " is histogram manipulation method A;" << endl;
	cout << "\t\t" << SUB_C_B_COMMAND << " is histogram manipulation method B;" << endl;
	cout << "\t\t" << "You do not need to input the size of the input image, because in this problem the input image is static." << endl;
	cout << "\t\t" << "Data of histograms and cumulative histograms will be print into standard output." << endl;
	cout << "\t\t" << "An output file \"" << OUTPUT_FILENAME << "\" will be create at the execution directory." << endl;
	return 0;
}

const int UNSIGNED_CHAR_MAX_VALUE = 0xFF;
const int COLOR_CHANNELS = 3;

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

inline float conv(unsigned char * data, const int dataHeight, const int dataWidth, const int row, const int col, float * filter, const int filterHeight, const int filterWidth) {
	int horiShift = filterWidth / 2;
	int vertShift = filterHeight / 2;
	float sum = 0;
	float weightSum = 0;
	for (int i = 0; i < filterHeight; i++) {
		for (int j = 0; j < filterWidth; j++) {
			float w = filter[flattenedIndex(filterWidth, i, j)];
			unsigned char d = data[flattenedIndex(dataWidth, reflectedIndex(dataHeight, row + i - vertShift), reflectedIndex(dataWidth, col + j - horiShift))];
			sum += w * d;
		}
	}
	return sum;
}

inline unsigned char roundAndClipToUnsignedChar(const float value) {
	//TODO: I don't know whether fmax() & fmin() functions are allowed here, so this implemention without min() & max() has branches. It is not friendly for branch prediction, although clip() is not needed here actually.
	int rounded = (int)round(value); //I use round() here to get a accurate approximation, while using round() or just casting away decimals visually looks the same.
	if (rounded > UNSIGNED_CHAR_MAX_VALUE) {
		return UNSIGNED_CHAR_MAX_VALUE;
	} else if (rounded < 0x00) {
		return 0x00;
	} else {
		return (unsigned char)rounded;
	}
}

void hist(unsigned char * data, const int size, float result[UNSIGNED_CHAR_MAX_VALUE + 1]) {
	memset(result, 0, (UNSIGNED_CHAR_MAX_VALUE + 1) * sizeof(float));
	for (int i = 0; i < size; i++) {
		result[data[i]]++;
	}
}

void normalize(float data[UNSIGNED_CHAR_MAX_VALUE + 1], const int size) {
	for (int i = 0; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
		data[i] /= size;
	}
}

void cumulativeDistribution(float histogram[UNSIGNED_CHAR_MAX_VALUE + 1], float result[UNSIGNED_CHAR_MAX_VALUE + 1]) {
	result[0] = histogram[0];
	for (int i = 1; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
		result[i] = histogram[i] + result[i - 1];
	}
}

template<typename T>
void printArray(T values[UNSIGNED_CHAR_MAX_VALUE + 1]) {
	cout << "{";
	for (int i = 0; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
		cout << (int)values[i] << ", ";
	}
	cout << "}" << endl;
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

/*=================================
|                                 |
|               (a)               |
|                                 |
=================================*/

//--------------(1)----------------

const int CAT_WIDTH = 390;
const int CAT_HEIGHT = 300;
const int BILINEAR_DEMOSAICING_FILTER_N = 3;

const float BL_CENTER_FILTER[BILINEAR_DEMOSAICING_FILTER_N][BILINEAR_DEMOSAICING_FILTER_N] = {
	{0.00, 0.00, 0.00, },
	{0.00, 1.00, 0.00, },
	{0.00, 0.00, 0.00, },
};

const float BL_DIAGONAL_FILTER[BILINEAR_DEMOSAICING_FILTER_N][BILINEAR_DEMOSAICING_FILTER_N] = {
	{0.25, 0.00, 0.25, },
	{0.00, 0.00, 0.00, },
	{0.25, 0.00, 0.25, },
};

const float BL_ADJACENT_FILTER[BILINEAR_DEMOSAICING_FILTER_N][BILINEAR_DEMOSAICING_FILTER_N] = {
	{0.00, 0.25, 0.00, },
	{0.25, 0.00, 0.25, },
	{0.00, 0.25, 0.00, },
};

const float BL_VERTICAL_FILTER[BILINEAR_DEMOSAICING_FILTER_N][BILINEAR_DEMOSAICING_FILTER_N] = {
	{0.00, 0.50, 0.00, },
	{0.00, 0.00, 0.00, },
	{0.00, 0.50, 0.00, },
};

const float BL_HORIZONTAL_FILTER[BILINEAR_DEMOSAICING_FILTER_N][BILINEAR_DEMOSAICING_FILTER_N] = {
	{0.00, 0.00, 0.00, },
	{0.50, 0.00, 0.50, },
	{0.00, 0.00, 0.00, },
};

void a(char* filename) {
	// Read image
	unsigned char input[CAT_HEIGHT][CAT_WIDTH];
	read(filename, input, CAT_WIDTH * CAT_HEIGHT);

	// Bilinear Demosaicing
	unsigned char output[CAT_HEIGHT][CAT_WIDTH][COLOR_CHANNELS];
	for (int i = 0; i < CAT_HEIGHT; i++) {
		for (int j = 0; j < CAT_WIDTH; j++) {
			int ii = i % 2;
			int jj = j % 2;
			if (ii == jj) { // Green
				output[i][j][0] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_HORIZONTAL_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//R
				output[i][j][1] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_CENTER_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//G
				output[i][j][2] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_VERTICAL_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//B
			} else if (ii == 0 && jj == 1) { // Red
				output[i][j][0] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_CENTER_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//R
				output[i][j][1] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_ADJACENT_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//G
				output[i][j][2] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_DIAGONAL_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//B
			} else { // Blue
				output[i][j][0] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_DIAGONAL_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//R
				output[i][j][1] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_ADJACENT_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//G
				output[i][j][2] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)BL_CENTER_FILTER, BILINEAR_DEMOSAICING_FILTER_N, BILINEAR_DEMOSAICING_FILTER_N));//B
			}
		}
	}

	// Write result
	write(output, CAT_WIDTH * CAT_HEIGHT * COLOR_CHANNELS, OUTPUT_FILENAME);
}

/*=================================
|                                 |
|               (b)               |
|                                 |
=================================*/

//--------------(1)----------------

const int MHC_DEMOSAICING_FILTER_SIZE = 5;

const float MHC_CENTER_FILTER[MHC_DEMOSAICING_FILTER_SIZE][MHC_DEMOSAICING_FILTER_SIZE] = {
	{+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, },
	{+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, },
	{+0.0000, +0.0000, +1.0000, +0.0000, +0.0000, },
	{+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, },
	{+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, },
};

const float MHC_CROSS_FILTER[MHC_DEMOSAICING_FILTER_SIZE][MHC_DEMOSAICING_FILTER_SIZE] = {
	{+0.0000, +0.0000, -0.1250, +0.0000, +0.0000, },
	{+0.0000, +0.0000, +0.2500, +0.0000, +0.0000, },
	{-0.1250, +0.2500, +0.5000, +0.2500, -0.1250, },
	{+0.0000, +0.0000, +0.2500, +0.0000, +0.0000, },
	{+0.0000, +0.0000, -0.1250, +0.0000, +0.0000, },
};

const float MHC_SQUARE_FILTER[MHC_DEMOSAICING_FILTER_SIZE][MHC_DEMOSAICING_FILTER_SIZE] = {
	{+0.0000, +0.0000, -0.1875, +0.0000, +0.0000, },
	{+0.0000, +0.2500, +0.0000, +0.2500, +0.0000, },
	{-0.1875, +0.0000, +0.7500, +0.0000, -0.1875, },
	{+0.0000, +0.2500, +0.0000, +0.2500, +0.0000, },
	{+0.0000, +0.0000, -0.1875, +0.0000, +0.0000, },
};

const float MHC_HORIZONTAL_FILTER[MHC_DEMOSAICING_FILTER_SIZE][MHC_DEMOSAICING_FILTER_SIZE] = {
	{+0.0000, +0.0000, +0.0625, +0.0000, +0.0000, },
	{+0.0000, -0.1250, +0.0000, -0.1250, +0.0000, },
	{-0.1250, +0.5000, +0.6250, +0.5000, -0.1250, },
	{+0.0000, -0.1250, +0.0000, -0.1250, +0.0000, },
	{+0.0000, +0.0000, +0.0625, +0.0000, +0.0000, },
};

const float MHC_VERTICAL_FILTER[MHC_DEMOSAICING_FILTER_SIZE][MHC_DEMOSAICING_FILTER_SIZE] = { // Transpose of HORIZONTAL_FILTER
	{+0.0000, +0.0000, -0.1250, +0.0000, +0.0000, },
	{+0.0000, -0.1250, +0.5000, -0.1250, +0.0000, },
	{+0.0625, +0.0000, +0.6250, +0.0000, +0.0625, },
	{+0.0000, -0.1250, +0.5000, -0.1250, +0.0000, },
	{+0.0000, +0.0000, -0.1250, +0.0000, +0.0000, },
};

void b(char* filename) {
	// Read image
	unsigned char input[CAT_HEIGHT][CAT_WIDTH];
	read(filename, input, CAT_WIDTH * CAT_HEIGHT);

	// Bilinear Demosaicing
	unsigned char output[CAT_HEIGHT][CAT_WIDTH][COLOR_CHANNELS];
	for (int i = 0; i < CAT_HEIGHT; i++) {
		for (int j = 0; j < CAT_WIDTH; j++) {
			int ii = i % 2;
			int jj = j % 2;
			if (ii == jj) { // Green
				output[i][j][0] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)(ii == 0 ? MHC_HORIZONTAL_FILTER : MHC_VERTICAL_FILTER), MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//R
				output[i][j][1] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_CENTER_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//G
				output[i][j][2] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)(ii == 0 ? MHC_VERTICAL_FILTER : MHC_HORIZONTAL_FILTER), MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//B
			} else if (ii == 0 && jj == 1) { // Red
				output[i][j][0] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_CENTER_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//R
				output[i][j][1] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_CROSS_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//G
				output[i][j][2] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_SQUARE_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//B
			} else { // Blue
				output[i][j][0] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_SQUARE_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//R
				output[i][j][1] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_CROSS_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//G
				output[i][j][2] = roundAndClipToUnsignedChar(conv((unsigned char *)input, CAT_HEIGHT, CAT_WIDTH, i, j, (float *)MHC_CENTER_FILTER, MHC_DEMOSAICING_FILTER_SIZE, MHC_DEMOSAICING_FILTER_SIZE));//B
			}
		}
	}

	// Write result
	write(output, CAT_WIDTH * CAT_HEIGHT * COLOR_CHANNELS, OUTPUT_FILENAME);
}

/*=================================
|                                 |
|               (c)               |
|                                 |
=================================*/

const int ROSE_WIDTH = 400;
const int ROSE_HEIGHT = 400;
const int ROSE_SIZE = ROSE_WIDTH * ROSE_HEIGHT;

//------------method A--------------

void ca(char* filename) {
	// Read image
	unsigned char input[ROSE_SIZE];
	read(filename, input, ROSE_SIZE);

	// Calculate histogram	
	float histogram[UNSIGNED_CHAR_MAX_VALUE + 1];
	hist(input, ROSE_SIZE, histogram);

	// Print input histogram (not necessary)
	cout << "Histogram data of the input image:" << endl;
	printArray(histogram);	
	
	// Calculate transfer-function	
	normalize(histogram, ROSE_SIZE);
	float cdf[UNSIGNED_CHAR_MAX_VALUE + 1];
	cumulativeDistribution(histogram, cdf);
	unsigned char mapping[UNSIGNED_CHAR_MAX_VALUE + 1];
	for (int i = 0; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
		mapping[i] = roundAndClipToUnsignedChar(cdf[i] * UNSIGNED_CHAR_MAX_VALUE);
	}
	cout << "Transfer function:" << endl;
	printArray(mapping);

	// Generate output data
	unsigned char output[ROSE_SIZE];
	for (int i = 0; i < ROSE_SIZE; i++) {
		output[i] = mapping[input[i]];
	}

	// Print output histogram (not necessary)
	hist(output, ROSE_SIZE, histogram);
	cout << "Histogram data of the output image:" << endl;
	printArray(histogram);

	// Write result
	write(output, ROSE_SIZE, OUTPUT_FILENAME);
}

//------------method B--------------

void cb(char* filename) {
	// Read image
	unsigned char input[ROSE_SIZE];
	read(filename, input, ROSE_SIZE);

	// Print input histogram (not necessary)
	float histogram[UNSIGNED_CHAR_MAX_VALUE + 1];
	hist(input, ROSE_SIZE, histogram);
	cout << "Histogram data of the input image:" << endl;
	printArray(histogram);
	float cdf[UNSIGNED_CHAR_MAX_VALUE + 1];
	cumulativeDistribution(histogram, cdf); // Unnormalized
	cout << "Cumulative histogram data of the input image:" << endl;
	printArray(cdf);

	// Generate output data
	int bucketSize = ROSE_SIZE / (UNSIGNED_CHAR_MAX_VALUE + 1) +  (ROSE_SIZE % (UNSIGNED_CHAR_MAX_VALUE + 1) > 0 ? 1 : 0);
	unsigned char output[ROSE_SIZE];
	int gray = 0;
	int count = 0;
	for (int i = 0; i <= UNSIGNED_CHAR_MAX_VALUE; i++) {
		for (int j = 0; j < ROSE_SIZE; j++) {
			if (input[j] == i) {
				output[j] = gray;
				count++;
				if (count >= bucketSize) {
					gray++;
					count = 0;
				}
			}
		}
	}

	// Print output histogram (not necessary)
	hist(output, ROSE_SIZE, histogram);
	cout << "Histogram data of the output image:" << endl;
	printArray(histogram);
	cumulativeDistribution(histogram, cdf); // Unnormalized
	cout << "Cumulative histogram data of the output image:" << endl;
	printArray(cdf);

	// Write result
	write(output, ROSE_SIZE, OUTPUT_FILENAME);
}

/*=================================
|                                 |
|              main               |
|                                 |
=================================*/
int main(int argc, char *argv[]) {
	if (argc < 3) {
		printUsage();
	} else if (strcmp(argv[1], SUB_A_COMMAND) == 0) {
		a(argv[2]);
	} else if (strcmp(argv[1], SUB_B_COMMAND) == 0) {
		b(argv[2]);
	} else if (strcmp(argv[1], SUB_C_A_COMMAND) == 0) {
		ca(argv[2]);
	} else if (strcmp(argv[1], SUB_C_B_COMMAND) == 0) {
		cb(argv[2]);
	} else {
		cout << "Wrong command!" << endl;
		printUsage();
	}
	return 0;
}
