%***********************************
%  Name: Zongjian Li               *
%  USC ID: 6503378943              *
%  USC Email: zongjian@usc.edu     *
%  Submission Date: 3rd, Mar 2019  *
%***********************************/

% This is the 2nd version of code of Problem 1 c. The 1st version is
% written in C++. Since C++ lacks of mathematical tools (and hard to 
% install and grade), the 1st version uses a algorithm does not need much 
% math operations.

function Problem1c
    clear;
    setEnvironment();
    
    global IMAGE_HEIGHT;
    global IMAGE_WIDTH;
    global CLASSROOM_IMAGE;
    newHeight = IMAGE_HEIGHT + 0;
    newWidth = IMAGE_WIDTH + 0;
    % Read
    in = readRaw(CLASSROOM_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH);
    linear = uint8(lensDistrotionCorrection_LinearRegression(in, IMAGE_HEIGHT, IMAGE_WIDTH, newHeight, newWidth));
    nonlinear = uint8(lensDistrotionCorrection_NonlinearRegression(in, IMAGE_HEIGHT, IMAGE_WIDTH, newHeight, newWidth));
    polar_nonlin = uint8(lensDistrotionCorrection_NonlinearRegression_Polar(in, IMAGE_HEIGHT, IMAGE_WIDTH, newHeight, newWidth));
    polar_lin = uint8(lensDistrotionCorrection_NonlinearRegression_Polar(in, IMAGE_HEIGHT, IMAGE_WIDTH, newHeight, newWidth));
    % Show results
    figure(1);
    imshow(in);
    figure(2);
    imshow(linear);
    figure(3);
    imshow(nonlinear);
    figure(4);
    imshow(polar_nonlin);
    figure(5);
    imshow(polar_lin);
    % Write file
    writeRaw(linear, 'linear.raw');
    writeRaw(nonlinear, 'nonlinear.raw');
    writeRaw(polar_nonlin, 'polar_nonlinear.raw');
    writeRaw(polar_lin, 'polar_linear.raw');
end

function setEnvironment
    clear all;
    close all;
    global IMAGE_HEIGHT;
    IMAGE_HEIGHT = 712;
    global IMAGE_WIDTH;
    IMAGE_WIDTH = 1072;
    global UINT8_MAX;
    UINT8_MAX = 255;
    global IMAGE_FOLDER;
    IMAGE_FOLDER = '../../Home Work No. 3/'; % Modify this!
    global CLASSROOM_IMAGE;
    CLASSROOM_IMAGE = [IMAGE_FOLDER '/' 'classroom.raw'];
    
    global K1;
    global K2;
    global F;
    K1 = -0.3536;
	K2 = 0.1730;
	F = 600;
end

function printError(message)
    error(message);
    exit(-1);
end

function result = readRaw(filename, height, width)
% Gray image version
    f = fopen(filename, 'rb');
    if (f == -1)
        printError('Can not open input image file');
    end
    [array, length] = fread(f, inf, 'uint8');
    fclose(f);
    if (length ~= height * width) 
        printError('size dismatch');
    end
    result = zeros(height, width, 'uint8');
    for i = 1 : height
        for j = 1 : width
            result(i, j) = array(width * (i - 1) + j);
        end
    end
    % imshow(image[0 255]);
end

function writeRaw(image, filename)
% Gray image version
    f = fopen(filename, 'wb');
    if (f == -1)
        printError('Can not open output image file');
    end
    fwrite(f, image', 'uint8');
    fclose(f);
end

function result = linearInterpolation(a, b, fa)
    if (fa < 0 || fa > 1)
        printError('ArgumentException');
    end
    result = a * fa + b * (1 - fa);
end

function result = bilinearInterpolation(image, row, col)
    rowLower = floor(row);
    rowHigher = ceil(row);
    colLower = floor(col);
    colHigher = ceil(col);
    valueLeftTop = image(rowLower, colLower);
    valueLeftBottom = image(rowHigher, colLower);
    valueRightTop = image(rowLower, colHigher);
    valueRightBottom = image(rowHigher, colHigher);
    lin1 = linearInterpolation(valueLeftTop, valueRightTop, 1 - (col - colLower));
    lin2 = linearInterpolation(valueLeftBottom, valueRightBottom, 1 - (col - colLower));
    result = linearInterpolation(lin1, lin2, 1 - (row - rowLower));
end

function [R, C, Rd, Cd] = mapping(height, width)
    global K1;
    global K2;
    global F;
    rowCenter = (height - 1) * 0.5;
	colCenter = (width - 1) * 0.5;
    R = zeros(height, width);
    C = zeros(height, width);
    Rd = zeros(height, width);
    Cd = zeros(height, width);
    for i = 0 : height - 1
        row = (i - rowCenter) / F;
        for j = 0 : width - 1
            col = (j - colCenter) / F;
            
            rSqr = row ^ 2 + col ^ 2;
			factor = 1 + K1 * rSqr + K2 * rSqr ^ 2;
			mappedRow = row * factor;
			mappedCol = col * factor;
            
            R(i + 1, j + 1) = row;
            C(i + 1, j + 1) = col;
            Rd(i + 1, j + 1) = mappedRow;
            Cd(i + 1, j + 1) = mappedCol;            
        end
    end
    R = reshape(R, height * width, 1);
    C = reshape(C, height * width, 1);
    Rd = reshape(Rd, height * width, 1);
    Cd = reshape(Cd, height * width, 1);
end

function result = linearModel(r, c)
    result = [ones(size(r, 1), 1), r, c, r .* c];
end

function [Rb, Cb] = linearRegression(height, width)
    [R, C, Rd, Cd] = mapping(height, width);
    X = linearModel(R, C);
    Rb = regress(Rd, X);
    Cb = regress(Cd, X);
end

function result = lensDistrotionCorrection_LinearRegression(input, height, width, newHeight, newWidth)
    [Rb, Cb] = linearRegression(newHeight, newWidth);
    global F;
    rowCenter = (height - 1) * 0.5;
    colCenter = (width - 1) * 0.5;
    newRowCenter = (newHeight - 1) * 0.5;
	newColCenter = (newWidth - 1) * 0.5;
    result = zeros(newHeight, newWidth);
    for i = 0 : newHeight - 1        
        row = (i - newRowCenter) / F;
        for j = 0 : newWidth - 1            
            col = (j - newColCenter) / F;
            distortedRow = Rb' * linearModel(row, col)';
            distortedRow = distortedRow * F + rowCenter;
            distortedCol = Cb' * linearModel(row, col)';           
            distortedCol = distortedCol * F + colCenter;
            
            if (0 <= distortedRow && distortedRow <= height - 1 && 0 <= distortedCol && distortedCol <= width - 1)
                result(i + 1, j + 1) = bilinearInterpolation(input, distortedRow + 1, distortedCol + 1);
            end
        end
    end
end

function result = nonlinearModel(a, x)
    r = x(:, 1);
    c = x(:, 2);
    f = [ones(size(x, 1), 1), ...
        r, c,  ...
        r .^ 2, r .* c, c .^ 2,  ...
        r .^ 3, r .^ 2 .* c, r .* c .^2, c .^ 3 ...
        r .^ 4, r .^ 3 .* c, r .^ 2 .* c .^ 2, r .* c .^ 3, c .^ 4 ...
        ];
    result = (a' * f')';
end

function result = nonlinearModelInit
	result = [ ...
        0; ...
        0; 0;  ...
        0; 0; 0;  ...
        0; 0; 0; 0; ...
        0; 0; 0; 0; 0 ...
        ];
end

function [Rb, Cb] = nonlinearRegression(height, width)
    [R, C, Rd, Cd] = mapping(height, width);
    X = [R, C];
    init = nonlinearModelInit();
    Rb = nlinfit(X, Rd, @nonlinearModel, init);
    Cb = nlinfit(X, Cd, @nonlinearModel, init);
end

function result = lensDistrotionCorrection_NonlinearRegression(input, height, width, newHeight, newWidth)
    [Rb, Cb] = nonlinearRegression(newHeight, newWidth);
    global F;
    rowCenter = (height - 1) * 0.5;
    colCenter = (width - 1) * 0.5;
    newRowCenter = (newHeight - 1) * 0.5;
	newColCenter = (newWidth - 1) * 0.5;
    result = zeros(newHeight, newWidth);
    for i = 0 : newHeight - 1        
        row = (i - newRowCenter) / F;
        for j = 0 : newWidth - 1     
            col = (j - newColCenter) / F;
            distortedRow = nonlinearModel(Rb, [row col]);
            distortedRow = distortedRow * F + rowCenter;
            distortedCol = nonlinearModel(Cb, [row col]);   
            distortedCol = distortedCol * F + colCenter;
            
            if (0 <= distortedRow && distortedRow <= height - 1 && 0 <= distortedCol && distortedCol <= width - 1)
                result(i + 1, j + 1) = bilinearInterpolation(input, distortedRow + 1, distortedCol + 1);
            end
        end
    end
end

function [R, Rd] = mapping_Polar(height, width)
    global K1;
    global K2;
    global F;
    rowCenter = (height - 1) * 0.5;
	colCenter = (width - 1) * 0.5;
    R = zeros(height, width);
    Rd = zeros(height, width);
    for i = 0 : height - 1
        row = (i - rowCenter) / F;
        for j = 0 : width - 1
            col = (j - colCenter) / F;
            
            rSqr = row ^ 2 + col ^ 2;
			factor = 1 + K1 * rSqr + K2 * rSqr ^ 2;
			mappedRow = row * factor;
			mappedCol = col * factor;
            
            R(i + 1, j + 1) = sqrt(rSqr);
            Rd(i + 1, j + 1) = sqrt(mappedRow ^ 2 + mappedCol ^ 2);          
        end
    end
    R = reshape(R, height * width, 1);
    Rd = reshape(Rd, height * width, 1);
end

function result = linearModel_Polar(r)
    result = [ones(size(r, 1), 1), r];
end

function Rb = linearRegression_Polar(height, width)
    [R, Rd] = mapping_Polar(height, width);
    X = linearModel_Polar(R, C);
    Rb = regress(Rd, X);
end

function result = lensDistrotionCorrection_LinearRegression_Polar(input, height, width, newHeight, newWidth)
    Rb = linearRegression_Polar(newHeight, newWidth);
    global F;
    rowCenter = (height - 1) * 0.5;
    colCenter = (width - 1) * 0.5;
    newRowCenter = (newHeight - 1) * 0.5;
	newColCenter = (newWidth - 1) * 0.5;
    result = zeros(newHeight, newWidth);
    for i = 0 : newHeight - 1        
        row = (i - newRowCenter) / F;
        for j = 0 : newWidth - 1            
            col = (j - newColCenter) / F;
            distortedRadius = Rb' * linearModel(sqrt(row ^ 2 + col ^ 2))';
            theta = atan2(row, col);
            distortedRow = distortedRadius * sin(theta);
            distortedRow = distortedRow * F + rowCenter;
            distortedCol = distortedRadius * cos(theta);
            distortedCol = distortedCol * F + colCenter;
            
            if (0 <= distortedRow && distortedRow <= height - 1 && 0 <= distortedCol && distortedCol <= width - 1)
                result(i + 1, j + 1) = bilinearInterpolation(input, distortedRow + 1, distortedCol + 1);
            end
        end
    end
end

function result = nonlinearModel_Polar(a, x)
    f = [ones(size(x, 1), 1), x, x .^ 2, x .^ 3, x .^ 4];
    result = (a' * f')';
end

function result = nonlinearModelInit_Polar
	result = [ ...
        0; ...
        0;  ...
        0;  ...
        0; ...
        0 ...
        ];
end

function Rb = nonlinearRegression_Polar(height, width)
    [R, Rd] = mapping_Polar(height, width);
    init = nonlinearModelInit_Polar();
    Rb = nlinfit(R, Rd, @nonlinearModel_Polar, init);
end

function result = lensDistrotionCorrection_NonlinearRegression_Polar(input, height, width, newHeight, newWidth)
    Rb = nonlinearRegression_Polar(newHeight, newWidth);
    global F;
    rowCenter = (height - 1) * 0.5;
    colCenter = (width - 1) * 0.5;
    newRowCenter = (newHeight - 1) * 0.5;
	newColCenter = (newWidth - 1) * 0.5;
    result = zeros(newHeight, newWidth);
    for i = 0 : newHeight - 1        
        row = (i - newRowCenter) / F;
        for j = 0 : newWidth - 1     
            col = (j - newColCenter) / F;
            distortedRadius = nonlinearModel_Polar(Rb, sqrt(row ^ 2 + col ^ 2));
            theta = atan2(row, col);
            distortedRow = distortedRadius * sin(theta);
            distortedRow = distortedRow * F + rowCenter;
            distortedCol = distortedRadius * cos(theta);
            distortedCol = distortedCol * F + colCenter;
            
            if (0 <= distortedRow && distortedRow <= height - 1 && 0 <= distortedCol && distortedCol <= width - 1)
                result(i + 1, j + 1) = bilinearInterpolation(input, distortedRow + 1, distortedCol + 1);
            end
        end
    end
end