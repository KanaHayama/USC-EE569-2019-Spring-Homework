%***********************************
%  Name: Zongjian Li               *
%  USC ID: 6503378943              *
%  USC Email: zongjian@usc.edu     *
%  Submission Date: 3rd, Mar 2019  *
%***********************************/

function minus
    height = 691;
    width = 550;
    defect = readRaw('deer.raw', height, width);
    defectless = readRaw('output.raw', height, width);
    result = zeros(size(defect), 'uint8');
    for i = 1 : height
        for j = 1 : width
            result(i, j) = defectless(i, j) - defect(i, j);
        end
    end
    imshow(result);
    writeRaw(result, 'deer_defect_area.raw');
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