%***********************************
%  Name: Zongjian Li               *
%  USC ID: 6503378943              *
%  USC Email: zongjian@usc.edu     *
%  Submission Date: 12th,Feb 2019  *
%***********************************/

function main
    clear;
    setEnvironment();
    global THRESHOLD_NUMBER;
    
%     % Try all thresholds and store all the results
%     result = analyze(thresholdNumber);
%     save(['results/all_t', num2str(THRESHOLD_NUMBER), '.mat'], 'result', '-v7.3');
    
    applyNewRequirementToAllImagesAndEdgeDetectors(THRESHOLD_NUMBER);
end

function setEnvironment
    clear all;
    close all;
    global IMAGE_HEIGHT;
    IMAGE_HEIGHT = 321;
    global IMAGE_WIDTH;
    IMAGE_WIDTH = 481;
    global COLOR_CHANNEL;
    COLOR_CHANNEL = 3;
    global GRAY_CHANNEL;
    GRAY_CHANNEL = 1;
    global UINT8_MAX;
    UINT8_MAX = 255;
    global GROUND_TRUTH_COUNT;
    GROUND_TRUTH_COUNT = 5;
    global IMAGE_FOLDER;
    IMAGE_FOLDER = '../../HW2_images/'; % Modify this!
    global PIG_IMAGE;
    PIG_IMAGE = [IMAGE_FOLDER, '/', 'Pig.raw'];
    global PIG_GROUND_TRUTH;
    PIG_GROUND_TRUTH = [IMAGE_FOLDER, '/', 'Pig.mat'];
    global TIGER_IMAGE;
    TIGER_IMAGE = [IMAGE_FOLDER, '/', 'Tiger.raw'];
    global TIGER_GROUND_TRUTH;
    TIGER_GROUND_TRUTH = [IMAGE_FOLDER, '/', 'Tiger.mat'];
    global HOUSE_IMAGE;
    HOUSE_IMAGE = [IMAGE_FOLDER, '/', 'House.png'];
    global PIOTR_TOOLBOX_FOLDER;
    PIOTR_TOOLBOX_FOLDER = './toolbox/'; % Modify this!
    global SE_TOOLBOX_FOLDER;
    SE_TOOLBOX_FOLDER = './edges/'; % Modify this!
    addpath(genpath(PIOTR_TOOLBOX_FOLDER));
    addpath(genpath(SE_TOOLBOX_FOLDER));   
    global THRESHOLD_NUMBER;
    THRESHOLD_NUMBER = 99; % Modify this!
end

function printError(message)
    error(message);
    exit(-1);
end

function result = reflectIndex(size, index)
    % index start from 1
    if (index < 1)
        result = -(index - 2);
    elseif (index > size)
        result = size - (index - size);
    else
        result = index;
    end        
end

function result = get(image, row, col)
    s = size(image);
    result = double(image(reflectIndex(s(1), row), reflectIndex(s(2), col)));
end

function result = convPixel(image, kernel, row, col)
    result = 0;
    vertShift = floor(size(kernel, 1) / 2);
    horiShift = floor(size(kernel, 2) / 2);
    for i = 1 : size(kernel, 1)
        for j = 1 : size(kernel, 2)
            factor = kernel(i, j);
            pixel = get(image, row - vertShift + i - 1, col - horiShift + j - 1);
            addition = factor * pixel;
            result = result + addition;
        end
    end    
end

function result = convAll(image, kernel)
    result = zeros(size(image), 'double');
    for i = 1 : size(image, 1)
        for j = 1 : size(image, 2)
            result(i, j) = convPixel(image, kernel, i, j);
        end
    end
end

function result = normalize(kernel)
    result = kernel;
    sum = 0;
    for i = 1 : size(kernel, 1)
       for j = 1 : size(kernel, 2)
           sum = sum + kernel(i, j);
       end
    end
    if (sum == 0)
        printError('All zeros kernel');
    end
    for i = 1 : size(kernel, 1)
       for j = 1 : size(kernel, 2)
           result = result / sum;
       end
    end
end

function result = gaussian(height, width, sigma)
    height = round(height);
    width = round(width);
    if (height < 1 || mod(height, 2) == 0 || width < 1 || mod(width, 2) == 0)
        printError('Invalid kernel size');
    end
    if (sigma <= 0)
        printError('Invalid sigma value');
    end
    filterVertMid = floor(height / 2) + 1;
    filterHoriMid = floor(width / 2) + 1;
    result = zeros(height, width);
    for i = 1 : height
        vDist = abs(i - filterVertMid);
        for j = 1 : width
            hDist = abs(j - filterHoriMid);
            result(i, j) = exp(-(vDist * vDist + hDist * hDist) / (2.0 * sigma * sigma));
        end
    end
    result = normalize(result);
end

function result = readRaw(filename, height, width, channel)
    f = fopen(filename, 'rb');
    if (f == -1)
        printError('Can not open input image file');
    end
    [array, length] = fread(f, inf, 'uint8');
    fclose(f);
    if (length ~= height * width * channel) 
        printError('size dismatch');
    end
    result = zeros(height, width, channel, 'uint8');
    for i = 1 : height
        for j = 1 : width
            for k = 1 : channel
                result(i, j, k) = array(channel * (width * (i - 1) + (j - 1)) + k);
            end
        end
    end
    % imshow(image,[0 255]);
end

function result = readPng(filename, height, width, channel)
    result = imread(filename);
    if (size(result, 1) ~= height || size(result, 2) ~= width || size(result, 3) ~= channel) 
        printError('size dismatch');
    end
end

function result = rgbToGray(input)
    result = rgb2gray(input); % Or using some other methods.    
end

function result = strech(array, upper)
    minValue = inf;
    maxValue = 0;
    for i = 1 : size(array, 1)
        for j = 1 : size(array, 2)
            minValue = min(minValue, array(i, j));
            maxValue = max(maxValue, array(i, j));
        end
    end
    shift = minValue;
    % shift = 0;
    factor = upper / (maxValue - shift);
    result = zeros(size(array), 'double');
    for i = 1 : size(array, 1)
        for j = 1 : size(array, 2)
            result(i, j) = (array(i, j) - shift) * factor;
        end
    end
end

function [result, percentageThreshold] = valueThresholding(gradient, threshold)
    gradient = strech(gradient, 1);
    result = zeros(size(gradient), 'logical');
    for i = 1 : size(result, 1)
        for j = 1 : size(result, 2)
            result(i, j) = gradient(i, j) > threshold;
        end
    end
    total = size(gradient, 1) * size(gradient, 2);
    list = sort(reshape(gradient, total, 1), 'descend');
    for i = 1 : length(list)
        if (list(i) < threshold)
            percentageThreshold = (i - 1) / total;
            return;
        end
    end
    percentageThreshold = 1;
end

function [result, valueThreshold] = percentageThresholding(gradient, threshold)    
    gradient = strech(gradient, 1);
    total = size(gradient, 1) * size(gradient, 2);
    edgeNumber = round(total * threshold);
    result = zeros(size(gradient), 'logical');
    valueThreshold = sort(reshape(gradient, total, 1), 'descend');
    valueThreshold = valueThreshold(edgeNumber);
%     if (valueThreshold == 0)
%         printError('No enough edge pixels to satisfiy edge percentage threshold');
%     end
   for i = 1 : size(gradient, 1)
       for j = 1 : size(gradient, 2)
           if (gradient(i, j) >= valueThreshold)
               result(i, j) = true;
           end           
       end
   end
end

function [gradient, xGradient, yGradient] = sobel(image)
    if (length(size(image)) == 3)
       image = rgbToGray(image); 
    end
    % imshow(image);
    xKernel = [1 0 -1; 2 0 -2; 1 0 -1];
    xGradient = convAll(image, xKernel);
    yKernel = [1 2 1; 0 0 0; -1 -2 -1];
    yGradient = convAll(image, yKernel);
    gradient = zeros(size(image));
    for i = 1 : size(gradient, 1)
        for j = 1 : size(gradient, 2)
            gradient(i, j) = sqrt(xGradient(i, j) * xGradient(i, j) + yGradient(i, j) * yGradient(i, j));
        end
    end
    global UINT8_MAX;
    gradient = strech(gradient, UINT8_MAX);
    xGradient = strech(xGradient, UINT8_MAX);
    yGradient = strech(yGradient, UINT8_MAX);
end

function result = canny(image, lower, higher)
    if (length(size(image)) == 3)
       image = rgbToGray(image); 
    end
    result = edge(image, 'Canny', [lower higher]);
end

function gradient = structured(image)    
    opts = edgesTrain();
    opts.modelDir = 'models/';
    opts.modelFnm = 'modelBsds';
    current = pwd;
    model = edgesTrain(opts); % load existing model, don't train it!    
    model.opts.multiscale = true; % Changed! The default option is false
    model.opts.sharpen = 2;
    model.opts.nTreesEval = 4;
    model.opts.nThreads = 4;
    model.opts.nms = true; % Changed! The default option is false
    gradient = edgesDetect(image, model);
    cd(current);
    global UINT8_MAX;
    gradient = strech(gradient, UINT8_MAX);    
end

function result = reverse(edge)
    result = zeros(size(edge), 'logical');
    for i = 1 : size(edge, 1)
        for j = 1 : size(edge, 2)
            result(i, j) = ~edge(i, j);
        end
    end
end

function result = sum(m)
    result = 0;
    for i = 1 : size(m)
       result = result + m(i); 
    end
end

function result = mean(m)
    result = sum(m) / size(m, 1);
end

function [meanP, meanR, F, P, R, V] = evaluate(groundTruthFilename, edge)
    g = load(groundTruthFilename);
    g = g.groundTruth;
    s = size(g, 2);
    P = zeros(s, 1);
    R = zeros(s, 1);
    for i = 1 : s
       groundTruth{1} = g{i};
       tempFilename = ['temp_ground_truth_file_', num2str(i), '.mat'];
       save(tempFilename, 'groundTruth');
       [~, cntR, sumR, cntP, sumP, v] = edgesEvalImg(edge, tempFilename, {'thrs', 1}); 
       delete(tempFilename);
       % imshow(v);
       V{i} = v;
       P(i) = cntP / sumP;
       R(i) = cntR / sumR;
    end
    meanP = mean(P);
    meanR = mean(R);   
    F = 2 * (meanP * meanR) / (meanP + meanR);
end

function result = analyze(thresholdNumber)
    global IMAGE_HEIGHT;
    global IMAGE_WIDTH;
    global COLOR_CHANNEL;
    global PIG_IMAGE;
    global PIG_GROUND_TRUTH;
    global TIGER_IMAGE;
    global TIGER_GROUND_TRUTH;
    pig = readRaw(PIG_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNEL);
    tiger = readRaw(TIGER_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNEL);
    thresholdInterval = 1 / (thresholdNumber + 1);
    thresholds = linspace(thresholdInterval,1 - thresholdInterval, thresholdNumber)';    
    imageCount = 2;
    methodCount = 3;
    result = cell(imageCount, methodCount, thresholdNumber, thresholdNumber);
    tic;
    for i = 1 : imageCount
        if (i == 1)
            input = pig;
            ground = PIG_GROUND_TRUTH;
        else
            input = tiger;
            ground = TIGER_GROUND_TRUTH;
        end        
        for m = 1 : methodCount
            for t1 = 1 : thresholdNumber                
                for t2 = 1 : thresholdNumber
                    result{i, m, t1, t2}.valid = false;
                    if (m == 1) % sobel
                        if (t1 ~= t2)
                            continue;
                        end
                        t = t1;
                        [gradient, xGradient, yGradient] = sobel(input); % you can save xGradient and yGradient here if you want.
                        [edge, percentageThreshold] = valueThresholding(gradient, thresholds(t));
                        if (percentageThreshold == 0 || percentageThreshold == 1)
                            continue;
                        end                        
                        result{i, m, t1, t2}.gradient = gradient;
                        result{i, m, t1, t2}.percentageThreshold = percentageThreshold;
                    elseif (m == 2) %canny
                        if (t1 >= t2) 
                            continue;
                        end
                        edge = canny(input, thresholds(t1), thresholds(t2));
                    else % structured
                        if (t1 ~= t2)
                            continue;
                        end
                        t = t1;
                        gradient = structured(input);  
                        [edge, percentageThreshold] = valueThresholding(gradient, thresholds(t));
                        if (percentageThreshold == 0 || percentageThreshold == 1)
                            continue;
                        end                                                                         
                        result{i, m, t1, t2}.gradient = gradient;
                        result{i, m, t1, t2}.percentageThreshold = percentageThreshold;
                        
                    end
                    [meanP, meanR, F, P, R, V] = evaluate(ground, edge);
                    display = reverse(edge);
                    
                    result{i, m, t1, t2}.edge = edge;
                    result{i, m, t1, t2}.display = display;
                    result{i, m, t1, t2}.meanP = meanP;
                    result{i, m, t1, t2}.meaR = meanR;
                    result{i, m, t1, t2}.F = F;
                    result{i, m, t1, t2}.P = P;
                    result{i, m, t1, t2}.R = R;
                    result{i, m, t1, t2}.V = V;
                    result{i, m, t1, t2}.valid = true;
                end
                ['Image ', num2str(i), ' method ', num2str(m), ' threshold ', num2str(thresholds(t1)), ' finished']
            end            
        end
    end
    toc;
end

function [Precision, Recall, mean_Precision_over_thresholds, mean_Recall_over_thresholds, mean_F, mean_Precision_overGTs, mean_Recall_over_GTs, F, thresholds] = newRequirement(thresholdNumber, method, image, percentage)
    % For newly posted requirement
    global IMAGE_HEIGHT;
    global IMAGE_WIDTH;
    global COLOR_CHANNEL;
    global PIG_IMAGE;
    global PIG_GROUND_TRUTH;
    global TIGER_IMAGE;
    global TIGER_GROUND_TRUTH;
    global GROUND_TRUTH_COUNT;
    if (strcmp(image, 'pig'))
        input = readRaw(PIG_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNEL);
        groundTruth = PIG_GROUND_TRUTH;
    elseif (strcmp(image, 'tiger'))
        input = readRaw(TIGER_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNEL);
        groundTruth = TIGER_GROUND_TRUTH;
    else
        printError('image argument should be pig or tiger.');
    end
    thresholdInterval = 1 / (thresholdNumber + 1);
    thresholds = linspace(thresholdInterval,1 - thresholdInterval, thresholdNumber)';   
    if (strcmp(method, 'sobel') || strcmp(method, 'structured'))
        Precision = zeros(GROUND_TRUTH_COUNT, thresholdNumber);
        Recall = zeros(GROUND_TRUTH_COUNT, thresholdNumber);
        mean_Precision_over_thresholds = zeros(GROUND_TRUTH_COUNT, 1);        
        mean_Recall_over_thresholds = zeros(GROUND_TRUTH_COUNT, 1);
        mean_Precision_overGTs = zeros(thresholdNumber, 1);
        mean_Recall_over_GTs  = zeros(thresholdNumber, 1);        
        totol_trail = thresholdNumber;
        for ti = 1 : thresholdNumber
            t = thresholds(ti);
            if (strcmp(method, 'sobel'))
                gradient = sobel(input);
            else
                gradient = structured(input);                  
            end
            if (percentage)
                edge = percentageThresholding(gradient, t);
            else
                edge = valueThresholding(gradient, t);
            end            
            [~, ~, ~, P, R, ~] = evaluate(groundTruth, edge);
            for g = 1 : GROUND_TRUTH_COUNT
                p = P(g);
                r = R(g);
                Precision(g, ti) = p;
                Recall(g, ti) = r;
                mean_Precision_over_thresholds(g) = mean_Precision_over_thresholds(g) + p / totol_trail;
                mean_Recall_over_thresholds(g) = mean_Recall_over_thresholds(g) + r / totol_trail;
                mean_Precision_overGTs(ti) = mean_Precision_overGTs(ti) + p / GROUND_TRUTH_COUNT;
                mean_Recall_over_GTs(ti) = mean_Recall_over_GTs(ti) + r / GROUND_TRUTH_COUNT;
            end            
        end
        F = zeros(thresholdNumber, 1);
        for ti = 1 : thresholdNumber
            F(ti) = 2 * mean_Precision_overGTs(ti) * mean_Recall_over_GTs(ti) / (mean_Precision_overGTs(ti) + mean_Recall_over_GTs(ti));
        end
    elseif (strcmp(method, 'canny'))
        Precision = zeros(GROUND_TRUTH_COUNT, thresholdNumber, thresholdNumber);
        Recall = zeros(GROUND_TRUTH_COUNT, thresholdNumber, thresholdNumber);
        mean_Precision_over_thresholds = zeros(GROUND_TRUTH_COUNT, 1);
        mean_Recall_over_thresholds = zeros(GROUND_TRUTH_COUNT, 1);
        mean_Precision_overGTs = zeros(thresholdNumber, thresholdNumber);
        mean_Recall_over_GTs  = zeros(thresholdNumber, thresholdNumber);   
        totol_trail = (thresholdNumber * thresholdNumber  - thresholdNumber) / 2;
        for ti = 1 : thresholdNumber
            t1 = thresholds(ti);
            for tj = 1 : thresholdNumber                    
                t2 = thresholds(tj);
                if (ti < tj)
                    edge = canny(input, t1, t2);
                    [~, ~, ~, P, R, ~] = evaluate(groundTruth, edge);
                    for g = 1 : GROUND_TRUTH_COUNT
                        p = P(g);
                        r = R(g);
                        Precision(g, ti, tj) = p;
                        Recall(g, ti, tj) = r;
                        mean_Precision_over_thresholds(g) = mean_Precision_over_thresholds(g) + p / totol_trail;
                        mean_Recall_over_thresholds(g) = mean_Recall_over_thresholds(g) + r / totol_trail;
                        mean_Precision_overGTs(ti, tj) = mean_Precision_overGTs(ti, tj) + p / GROUND_TRUTH_COUNT;
                        mean_Recall_over_GTs(ti, tj) = mean_Recall_over_GTs(ti, tj) + r / GROUND_TRUTH_COUNT;
                    end                    
                end                    
            end
        end
        F = zeros(thresholdNumber, thresholdNumber);
        for ti = 1 : thresholdNumber
            for tj = 1 : thresholdNumber
                if (ti < tj)
                    F(ti, tj) = 2 * mean_Precision_overGTs(ti, tj) * mean_Recall_over_GTs(ti, tj) / (mean_Precision_overGTs(ti, tj) + mean_Recall_over_GTs(ti, tj));
                end                
            end            
        end
    else
        printError('image method should be sobel or canny or structured.');
    end
    mean_F = 2 * mean(mean_Precision_over_thresholds) * mean(mean_Recall_over_thresholds) / (mean(mean_Precision_over_thresholds) + mean(mean_Recall_over_thresholds)); % This is my mean function, not the default one
end

function applyNewRequirementToAllImagesAndEdgeDetectors(thresholdNumber)        
    for imageI = 1 : 2
        switch imageI
           case 1
              image = 'pig';
           case 2
              image = 'tiger';
        end
        for methodI = 1 : 3
            switch methodI
            	case 1
                	method = 'sobel';
            	case 2
                	method = 'canny';
                case 3
                    method = 'structured';
            end 
            for percentage = 0 : 1
                if (percentage == 1)
                    type = 'percent';
                else
                    type = 'value';
                end
                if (methodI == 2 && percentage) % there is only one thresholding method for canny
                    continue;
                end
                [Precision, Recall, mean_Precision_over_thresholds, mean_Recall_over_thresholds, mean_F, mean_Precision_over_GTs, mean_Recall_over_GTs, F, thresholds] = newRequirement(thresholdNumber, method, image, percentage);
                if (strcmp(method, 'canny'))
                    graph = mesh(thresholds, thresholds, F);
                    bestF = max(max(F));
                else
                    graph = plot(thresholds, F);
                    bestF = max(F);
                end
                filename = ['requirement_result_', image, '_', method, '_thresholds_', type, '_', num2str(thresholdNumber), '.mat'];
                save(filename, 'image', 'method', 'Precision', 'Recall', 'mean_Precision_over_thresholds', 'mean_Recall_over_thresholds', 'mean_F', 'mean_Precision_over_GTs', 'mean_Recall_over_GTs', 'F', 'thresholds', 'graph', 'bestF');
                [filename, ' saved!']
            end            
        end
    end    
end