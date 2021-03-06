/************************************
 *  Name: Zongjian Li               *
 *  USC ID: 6503378943              *
 *  USC Email: zongjian@usc.edu     *
 *  Submission Date: 12th,Feb 2019  *
 ************************************/

 /*=================================
 |                                 |
 |              util               |
 |                                 |
 =================================*/

//#define  _CRT_SECURE_NO_WARNINGS 

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace std;

const char * ArgumentOutOfRangeException = "Argument out of range";

const char * ArgumentException = "Wrong argument";

const char * InvalidOperationException = "Invalid operation";

const char * FailedToOpenFileException = "Failed to open file";

const char * DEFAULT_OUTPUT_FILENAME = "output.raw";

const int UNSIGNED_CHAR_MAX_VALUE = 0xFF;

const int GRAY_CHANNELS = 1;

const int COLOR_CHANNELS = 3;

class Kernel {
	private:

	protected:
	int Height;
	int Width;
	vector<float> Data;

	int getSize() const {
		return Height * Width;
	}

	int getMemSize() const {
		return sizeof(float) * getSize();
	}

	int index(const int row, const int col) const {
		return row * Width + col;
	}	

	public:	

	Kernel() : Height(0), Width(0) {
		Data = vector<float>();
	}

	Kernel(const int height, const int width, const float & fill = 0) : Height(height), Width(width) {
		auto size = getSize();
		Data = vector<float>(size);
		for (int i = 0; i < size; i++) {
			Data[i] = fill;
		}
	}	

	Kernel(const int height, const int width, const float data[]) : Kernel(height, width) {		
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			Data[i] = data[i];
		}
	}

	Kernel operator * (const float f) const {
		auto result = Kernel(Height, Width);
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			result.Data[i] = Data[i] * f;
		}
		return result;
	}

	friend Kernel operator * (const float f, const Kernel & k);

	Kernel operator + (const float v) const {
		auto result = Kernel(Height, Width);
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			result.Data[i] = Data[i] + v;
		}
		return result;
	}

	friend Kernel operator + (const float v, const Kernel & k);

	Kernel operator / (const float f) const {
		auto result = Kernel(Height, Width);
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			result.Data[i] = Data[i] / f;
		}
		return result;
	}

	int getHeight() const {
		return Height;
	}

	int getWidth() const {
		return Width;
	}
	
	float getValue(const int row, const int col) const {
		if (row < 0 || row >= Height || col < 0 || col >= Width) {
			throw ArgumentOutOfRangeException;
		}
		return Data[index(row, col)];
	}

	void setValue(const int row, const int col, const float value) {
		if (row < 0 || row >= Height || col < 0 || col >= Width) {
			throw ArgumentOutOfRangeException;
		}
		Data[index(row, col)] = value;
	}

	void setValues(const int startRow, const int startCol, Kernel & values) {
		if (startRow < 0 || startRow + values.Height > Height || startCol < 0 || startCol + values.Width > Width) {
			throw ArgumentException;
		}
		for (int i = 0; i < values.Height; i++) {
			for (int j = 0; j < values.Width; j++) {
				setValue(startRow + i, startCol + j, values.getValue(i, j));
			}
		}
	}

	Kernel normalize() const {
		auto result = Kernel(Height, Width);
		double sum = 0;
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			sum += Data[i];
			result.Data[i] = Data[i];
		}
		if (sum != 0) {
			sum = sum > 0 ? sum : -sum;
			return result / sum;
		}		
		return result;
	}

	Kernel flipHorizontal() const {
		auto result = Kernel(Width, Height);
		for (int i = 0; i < Height; i++) {
			for (int j = 0; j < Width; j++) {
				result.setValue(i, Width - 1 - j, getValue(i, j));
			}
		}
		return result;
	}

	Kernel flipVertical() const {
		auto result = Kernel(Width, Height);
		for (int i = 0; i < Height; i++) {
			for (int j = 0; j < Width; j++) {				
				result.setValue(Height - 1 - i, j, getValue(i, j));
			}
		}
		return result;
	}

	~Kernel() {
		
	}
};

Kernel operator * (const float f, const Kernel & k) {
	return k * f;
}

Kernel operator + (const float v, const Kernel & k) {
	return k + v;
}

template <typename T>
class Image {
	private:

	protected:
	int Height;
	int Width;
	int Channel;
	vector<T> Data;

	int getSize() const {
		return Height * Width * Channel;
	}

	int getMemSize() const {
		return sizeof(unsigned char) * getSize();
	}

	static int reflect(const int size, const int index) {
		if (index <= -size || index >= size * 2 - 1) {
			throw ArgumentOutOfRangeException;
		}
		if (index < 0) {
			return -index;
		} else if (index >= size) {
			return 2 * size - index - 2;
		} else {
			return index;
		}
	}

	int index(const int row, const int col, const int ch) const {
		return Channel * (reflect(Height, row) * Width + reflect(Width, col)) + ch;
	}

	public:
	Image() : Height(0), Width(0), Channel(1) {
		Data = vector<T>();
	}

	Image(const int height, const int width, const int channel) : Height(height), Width(width), Channel(channel) {
		Data = vector<T>(getSize());
	}

	Image(const int height, const int width, const int channel, const T & fill) : Image(height, width, channel) {
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			Data[i] = fill;
		}
	}

	Image(const int height, const int width, const int channel, string filename) : Image(height, width, channel) {
		auto s = ifstream(filename, ifstream::binary);
		if (!s) {
			throw FailedToOpenFileException;
		}
		s.read((char *)&(Data[0]), getMemSize());
		s.close();
	}

	Image(const vector<Image> & channels) {
		if (channels.empty()) {
			throw ArgumentOutOfRangeException;
		}
		auto sample = channels.front();
		for (auto & ch : channels) {
			if (ch.Channel != 1) {
				throw ArgumentException;
			}
			if (ch.Width != sample.Width || ch.Height != sample.Height) {
				throw ArgumentException;
			}
		}
		Height = sample.Height;
		Width = sample.Width;
		Channel = channels.size();
		Data = vector<T>(getSize());
		for (int i = 0; i < Height; i++) {
			for (int j = 0; j < Width; j++) {
				for (int ch = 0; ch < Channel; ch++) {
					setValue(i, j, ch, channels[ch].getValue(i, j, 0));
				}
			}
		}
	}

	int getHeight() const {
		return Height;
	}

	int getWidth() const {
		return Width;
	}

	int getChannel() const {
		return Channel;
	}

	T getValue(const int row, const int col, const int ch, const bool enableReflect = true) const {
		if (!enableReflect && (row < 0 || row >= Height || col < 0 || col >= Width || ch < 0 || ch >= Channel)) {
			throw ArgumentOutOfRangeException;
		}
		return Data[index(row, col, ch)];
	}

	void setValue(const int row, const int col, const int ch, const T & value) {
		if (row < 0 || row >= Height || col < 0 || col >= Width || ch < 0 || ch >= Channel) {
			throw ArgumentOutOfRangeException;
		}
		Data[index(row, col, ch)] = value;
	}

	Image clip(const T lower, const T upper) const {
		auto result = Image(Height, Width, Channel);
		for (int i = 0; i < getSize(); i++) {
			result.Data[i] = min(upper, max(lower, Data[i]));
		}
	}

	Image round() const {
		auto result = Image(Height, Width, Channel);
		for (int i = 0; i < getSize(); i++) {
			result.Data[i] = int(Data[i] + 0.5);
		}
	}

	Image comp(const T upper = 0) const {
		auto result = Image(Height, Width, Channel);
		for (int i = 0; i < getSize(); i++) {
			result.Data[i] = upper - Data[i];
		}
		return result;
	}

	Image conv(const Kernel & kernel) const {
		if (kernel.getHeight() <= 0 || kernel.getWidth() <= 0) {
			throw ArgumentException;
		}
		if (kernel.getHeight() > Height || kernel.getWidth() > Width) {
			throw ArgumentException;
		}
		auto result = Image(Height, Width, Channel);
		auto horiShift = kernel.getWidth() / 2;
		auto vertShift = kernel.getHeight() / 2;
		for (int ch = 0; ch < Channel; ch++) { // ch = 0
			for (int i = 0; i < Height; i++) {
				for (int j = 0; j < Height; j++) {
					float sum = 0;
					for (int ii = 0; i < kernel.getHeight(); ii++) {
						auto inputRow = i + ii - vertShift;
						for (int jj = 0; j < kernel.getWidth(); jj++) {
							auto inputCol = j + jj - horiShift;
							sum += getValue(inputRow, inputCol, ch) * kernel.getValue(ii, jj);
						}
					}
					result.setValue(i, j, ch, sum); // TODO: round, if integer
				}
			}
		}
		return result;
	}

	vector<Image> split() const {
		auto result = vector<Image>(Channel);
		for (int ch = 0; ch < Channel; ch++) {
			result[ch] = Image(Height, Width, 1);
			for (int i = 0; i < Height; i++) {
				for (int j = 0; j < Width; j++) {
					result[ch].setValue(i, j, 0, getValue(i, j, ch));
				}
			}
		}
		return result;
	}

	void writeToFile(const string & filename) const {
		auto s = ofstream(filename, ofstream::binary);
		if (!s) {
			throw FailedToOpenFileException;
		}
		s.write((char *)&Data[0], getMemSize());
		s.close();
	}

	~Image() {
		
	}
};

const int DEFAULT_HEIGHT = 400;

const int DEFAULT_WIDTH = 600;

 /*=================================
 |                                 |
 |                a)               |
 |                                 |
 =================================*/

const float THRESHOLD = 0.5 * UNSIGNED_CHAR_MAX_VALUE;

Image<unsigned char> FixThresholding(const Image<unsigned char> & input) {
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), input.getChannel());
	for (int i = 0; i < result.getHeight(); i++) {
		for (int j = 0; j < result.getWidth(); j++) {
			for (int ch = 0; ch < result.getChannel(); ch++) {
				result.setValue(i, j, ch, input.getValue(i, j, ch) <= THRESHOLD ? 0 : UNSIGNED_CHAR_MAX_VALUE);
			}
		}
	}
	return result;
}

Image<unsigned char> RandomThresholding(const Image<unsigned char> & input, const int randomSeed = 0) {
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), input.getChannel());
	srand(randomSeed);
	for (int i = 0; i < result.getHeight(); i++) {
		for (int j = 0; j < result.getWidth(); j++) {
			for (int ch = 0; ch < result.getChannel(); ch++) {
				result.setValue(i, j, ch, input.getValue(i, j, ch) < (rand() % (UNSIGNED_CHAR_MAX_VALUE + 1)) ? 0 : UNSIGNED_CHAR_MAX_VALUE);
			}
		}
	}
	return result;
}

Kernel DitheringThresholds(const int power) {
	if (power <= 0) {
		throw ArgumentOutOfRangeException;
	}
	auto iter = power;
	auto size = 2;
	const float DITHERING_UNIT[] = {
		1, 2,
		3, 0,
	};
	auto result = Kernel(size, size, DITHERING_UNIT);
	iter--;
	while (iter > 0) {
		auto newSize = size * 2;		
		auto t4 = 4 * result;
		auto t4p1 = t4 + 1;
		auto t4p2 = t4 + 2;
		auto t4p3 = t4 + 3;
		result = Kernel(newSize, newSize);
		result.setValues(0, 0, t4p1);
		result.setValues(0, size, t4p2);
		result.setValues(size, 0, t4p3);
		result.setValues(size, size, t4);
		size = newSize;
		iter--;
	}
	return result;
}

Image<unsigned char> Dithering(const Image<unsigned char> & input, const int size) {
	if (size <= 0) {
		throw ArgumentOutOfRangeException;
	}
	auto power = log2(size);
	if (power - int(power) != 0.0) {
		throw ArgumentException;
	}
	auto kernel = DitheringThresholds(int(power));
	auto th = (kernel + 0.5) / (kernel.getHeight() * kernel.getWidth()) * 255;
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), input.getChannel());
	for (int i = 0; i < result.getHeight(); i++) {
		for (int j = 0; j < result.getWidth(); j++) {
			for (int ch = 0; ch < result.getChannel(); ch++) {
				result.setValue(i, j, ch, input.getValue(i, j, ch) < th.getValue(i % th.getHeight(), j % th.getWidth()) ? 0 : UNSIGNED_CHAR_MAX_VALUE); // TODO: should I round or trunc th to integers?
			}
		}
	}
	return result;
}

 /*=================================
 |                                 |
 |                b)               |
 |                                 |
 =================================*/

Kernel FloydSteinberg() {
	const float FS_UNIT[] = {
		0, 0, 0,
		0, 0, 7,
		3, 5, 1,
	};
	return Kernel(3, 3, FS_UNIT).normalize();
}

Kernel JarvisJudiceNinke() {
	const float JJN_UNIT[] = {
		0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0,
		0, 0, 0, 7, 5,
		3, 5, 7, 5, 3,
		1, 3, 5, 3, 1,
	};
	return Kernel(5, 5, JJN_UNIT).normalize();
}

Kernel Stucki() {
	const float S_UNIT[] = {
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 8, 4,
		2, 4, 8, 4, 2,
		1, 2, 4, 2, 1,
	};
	return Kernel(5, 5, S_UNIT).normalize();
}

Image<unsigned char> ErrorDiffusion(const Image<unsigned char> & input, const Kernel & kernel, const bool serpentine) {
	if (input.getChannel() != GRAY_CHANNELS) {
		throw ArgumentException;
	}
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), input.getChannel());
	auto additional = Image<float>(input.getHeight(), input.getWidth(), input.getChannel(), 0);
	auto filpKernel = kernel.flipHorizontal();
	auto vertShift = kernel.getHeight() / 2;
	auto horiShift = kernel.getWidth() / 2;
	for (int i = 0; i < result.getHeight(); i++) {
		auto reverse = serpentine && i % 2 == 0;
		auto begin = reverse ? 0 : result.getWidth() - 1;
		auto end = reverse ? result.getWidth() : -1;
		auto step = reverse ? 1 : -1;
		auto & k = reverse ? kernel : filpKernel;
		for (int j = begin; j != end; j += step) {
			for (int ch = 0; ch < result.getChannel(); ch++) {
				auto get = input.getValue(i, j, ch) + additional.getValue(i, j, ch);
				unsigned char set = get <= THRESHOLD ? 0 : UNSIGNED_CHAR_MAX_VALUE;
				auto diff = get - set;
				result.setValue(i, j, ch, set);
				for (int ii = 0; ii < k.getHeight(); ii++) {
					auto row = i + ii - vertShift;
					if (0 <= row && row < additional.getHeight()) {
						for (int jj = 0; jj < k.getWidth(); jj++) {
							auto col = j + jj - horiShift;
							if (0 <= col && col < additional.getWidth()) {
								auto f = k.getValue(ii, jj);
								additional.setValue(row, col, ch, additional.getValue(row, col, ch) + f * diff);
							}
						}
					}
				}
			}
		}
	}
	return result;
}

 /*=================================
 |                                 |
 |                c)               |
 |                                 |
 =================================*/

Image<unsigned char> SeparableErrorDiffusion(const Image<unsigned char> & input, const Kernel & kernel, const bool serpentine) {
	if (input.getChannel() != COLOR_CHANNELS) {
		throw ArgumentException;
	}
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), input.getChannel());
	auto cmy = input.comp(UNSIGNED_CHAR_MAX_VALUE); //Convert to CMY color space first!!
	auto additional = Image<float>(input.getHeight(), input.getWidth(), input.getChannel(), 0);
	auto filpKernel = kernel.flipHorizontal();
	auto vertShift = kernel.getHeight() / 2;
	auto horiShift = kernel.getWidth() / 2;
	for (int i = 0; i < result.getHeight(); i++) {
		auto reverse = serpentine && i % 2 == 0;
		auto begin = reverse ? 0 : result.getWidth() - 1;
		auto end = reverse ? result.getWidth() : -1;
		auto step = reverse ? 1 : -1;
		auto & k = reverse ? kernel : filpKernel;
		for (int j = begin; j != end; j += step) {
			for (int ch = 0; ch < result.getChannel(); ch++) {
				auto get = cmy.getValue(i, j, ch) + additional.getValue(i, j, ch);
				auto set = get <= THRESHOLD ? 0 : UNSIGNED_CHAR_MAX_VALUE;
				auto diff = get - set;
				result.setValue(i, j, ch, set);
				for (int ii = 0; ii < k.getHeight(); ii++) {
					auto row = i + ii - vertShift;
					if (0 <= row && row < additional.getHeight()) {
						for (int jj = 0; jj < k.getWidth(); jj++) {
							auto col = j + jj - horiShift;
							if (0 <= col && col < additional.getWidth()) {
								auto f = k.getValue(ii, jj);
								additional.setValue(row, col, ch, additional.getValue(row, col, ch) + f * diff);
							}
						}
					}
				}
			}
		}
	}
	return result.comp(UNSIGNED_CHAR_MAX_VALUE);
}

enum class Quadruples {
	KRGB,
	CMYW,
	MYGC,
	RGMY,
	RGBM,
	CMGB,
};

struct Pixel {
	unsigned char values[COLOR_CHANNELS];

	Pixel() {
		for (int i = 0; i < COLOR_CHANNELS; i++) {
			values[i] = 0;
		}
	}

	Pixel(const unsigned char c0, const unsigned char c1, const unsigned char c2) {
		values[0] = c0;
		values[1] = c1;
		values[2] = c2;
	}

	unsigned char getChannel(const int ch) const {
		if (ch < 0 || ch >= COLOR_CHANNELS) {
			throw ArgumentOutOfRangeException;
		}
		return values[ch];
	}

	void setChannel(const int ch, const int value) {
		if (ch < 0 || ch >= COLOR_CHANNELS) {
			throw ArgumentOutOfRangeException;
		}
		values[ch] = min(UNSIGNED_CHAR_MAX_VALUE, max(0, value));
	}

	unsigned char & operator [](const int ch) {
		return values[ch];
	}
};

struct RGB : public Pixel {
	RGB() : Pixel() {}

	RGB(const unsigned char red, const unsigned char green, const unsigned char blue) : Pixel(red, green, blue) {}

	unsigned char getRedChannel() const {
		return getChannel(0);
	}

	unsigned char getGreenChannel() const {
		return getChannel(1);
	}

	unsigned char getBlueChannel() const {
		return getChannel(2);
	}

	static RGB White() {
		return RGB(UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE);
	}

	static RGB Cyan() {
		return RGB(0, UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE);
	}

	static RGB Magenta() {
		return RGB(UNSIGNED_CHAR_MAX_VALUE, 0, UNSIGNED_CHAR_MAX_VALUE);
	}

	static RGB Yellow() {
		return RGB(UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE, 0);
	}

	static RGB Red() {
		return RGB(UNSIGNED_CHAR_MAX_VALUE, 0, 0);
	}

	static RGB Green() {
		return RGB(0, UNSIGNED_CHAR_MAX_VALUE, 0);
	}

	static RGB Blue() {
		return RGB(0, 0, UNSIGNED_CHAR_MAX_VALUE);
	}

	static RGB Black() {
		return RGB(0, 0, 0);
	}
};

struct CMY : public Pixel {
	CMY() : Pixel() {}

	CMY(const unsigned char cyan, const unsigned char magenta, const unsigned char yellow) : Pixel(cyan, magenta, yellow) {}

	CMY(const RGB & rgb) {
		for (int i = 0; i < COLOR_CHANNELS; i++) {
			values[i] = UNSIGNED_CHAR_MAX_VALUE - rgb.values[i];
		}
	}

	unsigned char getCyanChannel() const {
		return getChannel(0);
	}

	unsigned char getMagentaChannel() const {
		return getChannel(1);
	}

	unsigned char getYellowChannel() const {
		return getChannel(2);
	}

	static CMY White() {
		return CMY(0, 0, 0);
	}

	static CMY Cyan() {
		return CMY(UNSIGNED_CHAR_MAX_VALUE, 0, 0);
	}

	static CMY Magenta() {
		return CMY(0, UNSIGNED_CHAR_MAX_VALUE, 0);
	}

	static CMY Yellow() {
		return CMY(0, 0, UNSIGNED_CHAR_MAX_VALUE);
	}

	static CMY Red() {
		return CMY(0, UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE);
	}

	static CMY Green() {
		return CMY(UNSIGNED_CHAR_MAX_VALUE, 0, UNSIGNED_CHAR_MAX_VALUE);
	}

	static CMY Blue() {
		return CMY(UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE, 0);
	}

	static CMY Black() {
		return CMY(UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE, UNSIGNED_CHAR_MAX_VALUE);
	}
};

Quadruples SelectQuadruples(const RGB & rgb) {
	auto r = rgb.getRedChannel();
	auto g = rgb.getGreenChannel();
	auto b = rgb.getBlueChannel();
	if (r + g > UNSIGNED_CHAR_MAX_VALUE) {
		if (g + b > UNSIGNED_CHAR_MAX_VALUE) {
			if (r + g + b > 2 * UNSIGNED_CHAR_MAX_VALUE) {
				return Quadruples::CMYW;
			} else {
				return Quadruples::MYGC;
			}
		} else {
			return Quadruples::RGMY;
		}
	} else {
		if (g + b <= UNSIGNED_CHAR_MAX_VALUE) {
			if (r + g + b <= UNSIGNED_CHAR_MAX_VALUE) {
				return Quadruples::KRGB;
			} else {
				return Quadruples::RGBM;
			}
		} else {
			return Quadruples::CMGB;
		}
	}
}

RGB SelectColor(const RGB & rgb, const Quadruples quadruples) {
	auto r = rgb.getRedChannel();
	auto g = rgb.getGreenChannel();
	auto b = rgb.getBlueChannel();
	switch (quadruples) {
		case Quadruples::CMYW:
			if (b < THRESHOLD) {
				if (b < r) {
					if (b <= g) {
						return RGB::Yellow();
					}
				}
			}
			if (g < THRESHOLD) {
				if (g <= b) {
					if (g <= r) {
						return RGB::Magenta();
					}
				}
			}
			if (r < THRESHOLD) {
				if (r <= b) {
					if (r <= g) {
						return RGB::Cyan();
					}
				}
			}
			return RGB::White();
		case Quadruples::MYGC:
			if (g >= b) {
				if (r >= b) {
					if (r >= THRESHOLD) {
						return RGB::Yellow();
					} else {
						return RGB::Green();
					}
				}
			}
			if (g >= r) {
				if (b >= r) {
					if (b >= THRESHOLD) {
						return RGB::Cyan();
					} else {
						return RGB::Green();
					}
				}
			}
			return RGB::Magenta();
		case Quadruples::RGMY:
			if (b > THRESHOLD) {
				if (r > THRESHOLD) {
					if (b >= g) {
						return RGB::Magenta();
					} else {
						return RGB::Yellow();
					}
				} else {
					if (g > b + r) {
						return RGB::Green();
					} else {
						return RGB::Magenta();
					}
				}
			} else {
				if (r >= THRESHOLD) {
					if (g >= THRESHOLD) {
						return RGB::Yellow();
					} else {
						return RGB::Red();
					}
				} else {
					if (r >= g) {
						return RGB::Red();
					} else {
						return RGB::Green();
					}
				}
			}
		case Quadruples::KRGB:
			if (b > THRESHOLD) {
				if (b >= r) {
					if (b >= g) {
						return RGB::Blue();
					}
				}
			}
			if (g > THRESHOLD) {
				if (g > b) {
					if (g > r) {
						return RGB::Green();
					}
				}
			}
			if (r > THRESHOLD) {
				if (r >= b) {
					if (r >= g) {
						return RGB::Red();
					}
				}
			}
			return RGB::Black();
		case Quadruples::RGBM:
			if (r > g) {
				if (r >= b) {
					if (b < THRESHOLD) {
						return RGB::Red();
					} else {
						return RGB::Magenta();
					}
				}
			}
			if (b > g) {
				if (b >= r) {
					if (r < THRESHOLD) {
						return RGB::Blue();
					} else {
						return RGB::Magenta();
					}
				}
			}
			return RGB::Green();
		case Quadruples::CMGB:
			if (b > THRESHOLD) {
				if (r > THRESHOLD) {
					if (g >= r) {
						return RGB::Cyan();
					} else {
						return RGB::Magenta();
					}
				} else {
					if (g > THRESHOLD) {
						return RGB::Cyan();
					} else {
						return RGB::Blue();
					}
				}
			} else {
				if (r > THRESHOLD) {
					if (r - g + b >= THRESHOLD) {
						return RGB::Magenta();
					} else {
						return RGB::Green();
					}
				} else {
					if (g >= b) {
						return RGB::Green();
					} else {
						return RGB::Blue();
					}
				}
			}
	}
}

Image<unsigned char> MBVQ(const Image<unsigned char> & input, const Kernel & kernel, const bool serpentine) {
	if (input.getChannel() != COLOR_CHANNELS) {
		throw ArgumentException;
	}
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), input.getChannel());
	auto additional = Image<float>(input.getHeight(), input.getWidth(), input.getChannel(), 0);
	auto filpKernel = kernel.flipHorizontal();
	auto vertShift = kernel.getHeight() / 2;
	auto horiShift = kernel.getWidth() / 2;
	for (int i = 0; i < result.getHeight(); i++) {
		auto reverse = serpentine && i % 2 == 0;
		auto begin = reverse ? 0 : result.getWidth() - 1;
		auto end = reverse ? result.getWidth() : -1;
		auto step = reverse ? 1 : -1;
		auto & k = reverse ? kernel : filpKernel;
		for (int j = begin; j != end; j += step) {
			auto inputPixel = RGB();
			auto adjustedPixel = RGB();
			for (int ch = 0; ch < result.getChannel(); ch++) {
				inputPixel.setChannel(ch, round(input.getValue(i, j, ch)));
				adjustedPixel.setChannel(ch, round(input.getValue(i, j, ch) + additional.getValue(i, j, ch)));
			}
			auto quadruples = SelectQuadruples(inputPixel); // modified base on a discussion on den
			auto outputPixel = SelectColor(adjustedPixel, quadruples);
			for (int ch = 0; ch < result.getChannel(); ch++) {
				auto get = adjustedPixel.getChannel(ch);
				auto set = outputPixel.getChannel(ch);
				auto diff = get - set;
				result.setValue(i, j, ch, set);
				for (int ii = 0; ii < k.getHeight(); ii++) {
					auto row = i + ii - vertShift;
					if (0 <= row && row < additional.getHeight()) {
						for (int jj = 0; jj < k.getWidth(); jj++) {
							auto col = j + jj - horiShift;
							if (0 <= col && col < additional.getWidth()) {
								auto f = k.getValue(ii, jj);
								additional.setValue(row, col, ch, additional.getValue(row, col, ch) + f * diff);
							}
						}
					}
				}
			}
		}
	}
	return result;
}

 /*=================================
 |                                 |
 |              main               |
 |                                 |
 =================================*/

const char * OPTION_METHOD = "-a";
const char * OPTION_RANDOM_SEED = "-s";
const char * OPTION_DITHERING_SIZE = "-d";
const char * OPTION_MATRIX = "-m";
const char * OPTION_OUTPUT = "-o";
const char * OPTION_HEIGHT = "-h";
const char * OPTION_WIDTH = "-w";
const char * OPTION_CHANNEL = "-c";
const char * OPTION_SCAN_ORDER = "-so";

const char * METHOD_FIX_THRESHOLDING = "ft";
const char * METHOD_RANDOM_THRESHOLDING = "rt";
const char * METHOD_DITHERING = "d";
const char * METHOD_ERROR_DIFFUSION = "ed";
const char * METHOD_SEPARABLE_ERROR_DIFFUSION = "sed";
const char * METHOD_MBVQ_BASED_ERROR_DIFFUSION = "mbvq";

const char * MATRIX_FLOYD_STEINBERG = "fs";
const char * MATRIX_JARVIS_JUDICE_NINKE = "jjn";
const char * MATRIX_STUCKI = "s";

const char * SCAN_ORDER_RASTER = "r";
const char * SCAN_ORDER_SERPENTINE = "s";

enum class MethodType {
	FixThresholding,
	RandomThresholding,
	Dithering,
	ErrorDiffusion,
	SeparableErrorDiffusion,
	MbvqBasedErrorDiffusion,
};

enum class MatrixType {
	FloydSteinberg,
	JarvisJudiceNinke,
	Stucki,
};

enum class ScanOrder {
	Raster,
	Serpentine,
};

const MethodType DefaultMethod = MethodType::RandomThresholding;
const int DefaultRandomSeed = 0;
const int DefaultDitheringSize = 32;
const MatrixType DefaultMatrix = MatrixType::FloydSteinberg;
const ScanOrder DefaultScanOrder = ScanOrder::Serpentine;

const char * WrongCommandException = "Wrong command";

MethodType ParseMethod(const string & cmd) {
	if (cmd == METHOD_FIX_THRESHOLDING) {
		return MethodType::FixThresholding;
	}else if (cmd == METHOD_RANDOM_THRESHOLDING) {
		return MethodType::RandomThresholding;
	} else if (cmd == METHOD_DITHERING) {
		return MethodType::Dithering;
	} else if (cmd == METHOD_ERROR_DIFFUSION) {
		return MethodType::ErrorDiffusion;
	} else if (cmd == METHOD_SEPARABLE_ERROR_DIFFUSION) {
		return MethodType::SeparableErrorDiffusion;
	} else if (cmd == METHOD_MBVQ_BASED_ERROR_DIFFUSION) {
		return MethodType::MbvqBasedErrorDiffusion;
	} else {
		throw WrongCommandException;
	}
}

MatrixType ParseMatrix(const string & cmd) {
	if (cmd == MATRIX_FLOYD_STEINBERG) {
		return MatrixType::FloydSteinberg;
	} else if (cmd == MATRIX_JARVIS_JUDICE_NINKE) {
		return MatrixType::JarvisJudiceNinke;
	} else if (cmd == MATRIX_STUCKI) {
		return MatrixType::Stucki;
	} else {
		throw WrongCommandException;
	}
}

ScanOrder ParseScanOrder(const string & cmd) {
	if (cmd == SCAN_ORDER_RASTER) {
		return ScanOrder::Raster;
	} else if (cmd == SCAN_ORDER_SERPENTINE) {
		return ScanOrder::Serpentine;
	} else {
		throw WrongCommandException;
	}
}

void PrintUsage() {
	cerr << "Usage:" << endl
		<< "\t" << "Problem2 [OPTION]... [INPUT FILE]" << endl
		<< endl
		<< "Intro:" << endl
		<< "\t" << "Digital image half-toning." << endl
		<< "\t" << "For USC EE569 2019 spring home work 2 problem 2 by Zongjian Li." << endl
		<< endl
		<< "Options:" << endl
		<< "\t" << OPTION_METHOD << "\t" << "Half-toning method." << endl
		<< "\t\t" << "You can choose from \"" << METHOD_FIX_THRESHOLDING << "\"(Fix Thresholding), \"" << METHOD_RANDOM_THRESHOLDING << "\"(Random Thresholding), \"" << METHOD_DITHERING << "\"(Dithering), \"" << METHOD_ERROR_DIFFUSION << "\"(Error Diffusion), \"" << METHOD_SEPARABLE_ERROR_DIFFUSION << "\"(Separable Error Diffusion), \"" << METHOD_MBVQ_BASED_ERROR_DIFFUSION << "\"(MBVQ-based Error Diffusion)." << endl
		<< "\t\t" << "The default is Random Thresholding." << endl
		<< "\t" << OPTION_RANDOM_SEED << "\t" << "Randm seed for Random Thresholding method. The default is " << DefaultRandomSeed << "." << endl
		<< "\t" << OPTION_DITHERING_SIZE << "\t" << "Height / Width of dithering matrix used in Dithering method. Must be a power of 2, smaller than input image size. The default is " << DefaultDitheringSize << "." << endl
		<< "\t" << OPTION_MATRIX << "\t" << "Error diffusion matrix." << endl
		<< "\t\t" << "You can choose from \"" << MATRIX_FLOYD_STEINBERG << "\"(Floyd-Steinberg's), \"" << MATRIX_JARVIS_JUDICE_NINKE << "\"(Jarvis, Judice, and Ninke (JJN)), \"" << MATRIX_STUCKI << "\"(Stucki)." << endl
		<< "\t\t" << "The default is Floyd-Steinberg's." << endl
		<< "\t" << OPTION_SCAN_ORDER << "\t" << "Scanning Order." << endl
		<< "\t\t" << "You can choose from \"" << SCAN_ORDER_RASTER << "\"(Raster), \"" << SCAN_ORDER_SERPENTINE << "\"(Serpentine)." << endl
		<< "\t\t" << "The default is Serpentine." << endl
		<< "\t" << OPTION_OUTPUT << "\t" << "Output filename. The default is \"" << DEFAULT_OUTPUT_FILENAME << "\"." << endl
		<< "\t" << OPTION_HEIGHT << "\t" << "Height of the input image. The default is " << DEFAULT_HEIGHT << "." << endl
		<< "\t" << OPTION_WIDTH << "\t" << "Width of the input image. The default is " << DEFAULT_WIDTH << "." << endl
		<< "\t" << OPTION_CHANNEL<< "\t" << "Number of channels of the input image. The default is " << GRAY_CHANNELS << "." << endl
		<< endl
		<< "Example:" <<endl
		<< "\t" << "Problem2 -a " << METHOD_MBVQ_BASED_ERROR_DIFFUSION << " -o my_output_image.raw my_input_image.raw" << endl 
		<< endl;
}

int main(int argc, char *argv[]) {
	auto method = DefaultMethod;
	auto randomSeed = DefaultRandomSeed;
	auto ditheringSize = DefaultDitheringSize;
	auto matrix = DefaultMatrix;
	auto order = DefaultScanOrder;
	auto output = string(DEFAULT_OUTPUT_FILENAME);
	auto height = DEFAULT_HEIGHT;
	auto width = DEFAULT_WIDTH;
	auto channel = GRAY_CHANNELS;
	auto methodFlag = false;
	auto randomSeedFlag = false;
	auto ditheringSizeFlag = false;
	auto matrixFlag = false;
	auto orderFlag = false;
	auto outputFlag = false;
	auto heightFlag = false;
	auto widthFlag = false;
	auto channelFlag = false;
	auto input = string();
	try {
		int i;
		for (i = 1; i < argc; i++) {
			auto cmd = string(argv[i]);
			if (methodFlag) {
				method = ParseMethod(cmd);
				methodFlag = false;
			} else if (randomSeedFlag) {
				randomSeed = atoi(cmd.c_str());
				randomSeedFlag = false;
			} else if (ditheringSizeFlag) {
				ditheringSize = atoi(cmd.c_str());
				ditheringSizeFlag = false;
			} else if (matrixFlag) {
				matrix = ParseMatrix(cmd);
				matrixFlag = false;
			} else if (orderFlag) {
				order = ParseScanOrder(cmd);
				orderFlag = false;
			} else if (outputFlag) {
				output = cmd;
				outputFlag = false;
			} else if (heightFlag) {
				height = atoi(cmd.c_str());
				heightFlag = false;
			} else if (widthFlag) {
				width = atoi(cmd.c_str());
				widthFlag = false;
			} else if (channelFlag) {
				channel = atoi(cmd.c_str());
				channelFlag = false;
			} else if (cmd == OPTION_METHOD) {
				methodFlag = true;
			} else if (cmd == OPTION_RANDOM_SEED) {
				randomSeedFlag = true;
			} else if (cmd == OPTION_DITHERING_SIZE) {
				ditheringSizeFlag = true;
			} else if (cmd == OPTION_MATRIX) {
				matrixFlag = true;
			} else if (cmd == OPTION_SCAN_ORDER) {
				orderFlag = true;
			} else if (cmd == OPTION_OUTPUT) {
				outputFlag = true;
			} else if (cmd == OPTION_HEIGHT) {
				heightFlag = true;
			} else if (cmd == OPTION_WIDTH) {
				widthFlag = true;
			} else if (cmd == OPTION_CHANNEL) {
				channelFlag = true;
			} else {
				input = cmd;
				break;
			}			
		}
		if (input == "" || i != argc - 1 || methodFlag || randomSeedFlag || ditheringSizeFlag || matrixFlag || orderFlag || outputFlag || heightFlag || widthFlag || channelFlag) {
			PrintUsage();
			throw WrongCommandException;
		}
		auto in = Image<unsigned char>(height, width, channel, input);
		//in.writeToFile("in.raw");
		auto kernel = Kernel();
		switch (matrix) {
			case MatrixType::FloydSteinberg:
				kernel = FloydSteinberg();
				break;
			case MatrixType::JarvisJudiceNinke:
				kernel = JarvisJudiceNinke();
				break;
			case MatrixType::Stucki:
				kernel = Stucki();
				break;
		}
		Image<unsigned char> out;
		switch (method) {
			case MethodType::FixThresholding:
				out = FixThresholding(in);
				break;
			case MethodType::RandomThresholding:
				out = RandomThresholding(in, randomSeed);
				break;
			case MethodType::Dithering:
				out = Dithering(in, ditheringSize);
				break;
			case MethodType::ErrorDiffusion:
				out = ErrorDiffusion(in, kernel, order == ScanOrder::Serpentine);
				break;
			case MethodType::SeparableErrorDiffusion:
				out = SeparableErrorDiffusion(in, kernel, order == ScanOrder::Serpentine);
				break;
			case MethodType::MbvqBasedErrorDiffusion:
				out = MBVQ(in, kernel, order == ScanOrder::Serpentine);
				break;
		}
		out.writeToFile(output);
		return 0;
	} catch (const char * ex) {
		cerr << "Captured exception: " << ex << endl;
	}	
	return 1;
}