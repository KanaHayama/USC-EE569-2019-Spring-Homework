%***********************************
%  Name: Zongjian Li               *
%  USC ID: 6503378943              *
%  USC Email: zongjian@usc.edu     *
%  Submission Date: 19th,Mar 2019  *
%***********************************/

function Problem1
    setEnvironment();
    
    figure(1);
    problem1a();
    
    figure(2);
    problem1b();
end

function setEnvironment
    clc;
    clear all;
    close all;
    global IMAGE_HEIGHT;
    IMAGE_HEIGHT = 128;
    global IMAGE_WIDTH;
    IMAGE_WIDTH = 128;
    global COLOR_CHANNEL;
    COLOR_CHANNEL = 3;
    global GRAY_CHANNEL;
    GRAY_CHANNEL = 1;
    global UINT8_MAX;
    UINT8_MAX = 255;
    global IMAGE_FOLDER;
    IMAGE_FOLDER = '../../HW4_images/'; % Modify this!
    global IMAGE_PREFIX;
    IMAGE_PREFIX = 'texture';
    global IMAGE_SUFIX;
    IMAGE_SUFIX = '.raw';
    global IMAGE_COUNT;
    IMAGE_COUNT = 12;
    global COMB_FILENAME;
    COMB_FILENAME = 'comb.raw';
    global COMB_HEIGHT;
    COMB_HEIGHT = 510;
    global COMB_WIDTH;
    COMB_WIDTH = 510;
    global CLASSIFY_NUM;
    CLASSIFY_NUM = 4;
    global SEGMENT_NUM;
    SEGMENT_NUM = 7;
    global MINE_KMEANS;
    MINE_KMEANS = false;
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

function result = ReadRaw(filename, height, width, channel)
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

function WriteRaw(image, filename)
    f = fopen(filename, 'wb');
    if (f == -1)
        printError('Can not open output image file');
    end
    fwrite(f, image', 'uint8');
    fclose(f);
end

function result = Shrink(image)
    global UINT8_MAX;
    result = zeros(size(image), 'double');
    for i = 1 : size(image, 1)
        for j = 1 : size(image, 2)
            result(i, j) = double(image(i, j)) / double(UINT8_MAX);
        end
    end
end

function result = Subtract(a, b)
    if size(a) ~= size(b)
        printError("Size dismatch");
    end
    result = zeros(size(a));
    for i = 1 : size(a, 1)
        for j = 1 : size(a, 2)
            result(i, j) = a(i, j) - b(i, j);
        end
    end
end

%***********************************
%               laws               *
%***********************************/

function result = LocalMeans(image, windowSize)
    meanKernal = fspecial('average', windowSize);
    result = convAll(image, meanKernal);
end

function result = SubtractLocalMeans(image, windowSize)
    localMeans = LocalMeans(image, windowSize);
    result = Subtract(image, localMeans);
end

function result = GenLawsFilter(a, b)
    result = zeros(5, 5);
    for i = 1 : 5
        for j = 1 : 5
           result(i, j) = a(i) * b(j); 
        end
    end
end

function result = LawsFilters()
    laws = [
        [1 4 6 4 1] %L5
        [-1 -2 0 2 1] %E5
        [-1 0 2 0 -1] %S5
        [-1 2 0 -2 1] %W5
        [1 -4 6 -4 1] %R5
        ];
    result = cell(5, 5);
    for i = 1 : 5
        for j = 1 : 5
            result{i, j} = GenLawsFilter(squeeze(laws(i, :)), squeeze(laws(j, :)));
        end
    end
end

function result = ApplyLawsFilters(image)
    filters = LawsFilters();
    result = cell(5, 5);
    for i = 1 : 5
        for j = 1 : 5
            result{i, j} = convAll(image, filters{i, j});
        end
    end
end

function result = PositiveEnergies(images)
    result = cell(5, 5);
    for i = 1 : 5
        for j = 1 : 5
            result{i, j} = images{i, j} .^ 2;%abs(images{i, j});
        end
    end
end

function result = AverageEnergy(matrices)
    height = size(matrices{1, 1}, 1);
    width = size(matrices{1, 1}, 2);
    result = zeros(5, 5);
    for i = 1 : 5
        for j = 1 : 5
            for ii = 1 : height
                for jj = 1 : width
                    result(i, j) = result(i, j) + matrices{i, j}(ii, jj);
                end
            end
            result(i, j) = result(i, j) ./ (height * width);
        end
    end
end

function result = AverageEnergies(matrices, windowSize)
    meanKernal = fspecial('average', windowSize);
    height = size(matrices{1, 1}, 1);
    width = size(matrices{1, 1}, 2);
    result = cell(height, width);
    for i = 1 : height
        for j = 1 : width
            result{i, j} = zeros(5, 5);
            for ii = 1 : 5
                for jj = 1 : 5
                    result{i, j}(ii, jj) = convPixel(matrices{ii, jj}, meanKernal, i, j);
                end
            end
        end
    end
end

function result = ReshapeMatrixToVector(matrix)
    result = reshape(matrix, 5 * 5, 1);
end

function result = ReshapeMatricesToVectors(matrices)
    height = size(matrices, 1);
    width = size(matrices, 2);
    result = cell(height, width);
    for i = 1 : height
        for j = 1 : width
            result{i, j} = ReshapeMatrixToVector(matrices{i, j});
        end
    end
end

function result = Reshape15DimVector(vector)
    vector = reshape(vector, 5, 5);
    result = zeros(15, 1);
    result(1) = vector(1, 1); %LL
    result(2) = vector(2, 2); %EE
    result(3) = vector(3, 3); %SS
    result(4) = vector(4, 4); %WW
    result(5) = vector(5, 5); %RR
    result(6) = (vector(1, 2) + vector(2, 1)) / 2; %LE/EL
    result(7) = (vector(1, 3) + vector(3, 1)) / 2; %LS/SL
    result(8) = (vector(1, 4) + vector(4, 1)) / 2; %LW/WL
    result(9) = (vector(1, 5) + vector(5, 1)) / 2; %LR/RL
    result(10) = (vector(4, 5) + vector(5, 4)) / 2; %WR/RW
    result(11) = (vector(2, 3) + vector(3, 2)) / 2; %ES/SE
    result(12) = (vector(2, 4) + vector(4, 2)) / 2; %EW/WE
    result(13) = (vector(2, 5) + vector(5, 2)) / 2; %ER/RE
    result(14) = (vector(3, 4) + vector(4, 3)) / 2; %SW/WS
    result(15) = (vector(3, 5) + vector(5, 3)) / 2; %SR/RS
end

function result = CombineSymmetricPairs(vector, flag)
    if flag
        result = Reshape15DimVector(vector);
    else
        result = vector;
    end
end

function result = CombineAllSymmetricPairs(vectors, flag)
    height = size(vectors, 1);
    width = size(vectors, 2);
    result = cell(height, width);
    for i = 1 : height
        for j = 1 : width
            result{i, j} = CombineSymmetricPairs(vectors{i, j}, flag);
        end
    end
end

function result = Normalize(vector)
    result = zeros(size(vector, 1), 1);
    L5L5 = vector(1); %L5L5
    for k = 1 : size(vector, 1)
        result(k) = vector(k) / L5L5;
    end
end

function result = NormalizeAll(vectors)
    height = size(vectors, 1);
    width = size(vectors, 2);
    result = cell(height, width);
    for i = 1 : height
        for j = 1 : width
            result{i, j} = Normalize(vectors{i, j});
        end
    end
end

function result = ReduceDimension(fullMatrix, dim)
    if dim > size(fullMatrix, 2)
        printError("Wrong dim");
    elseif dim == size(fullMatrix, 2)
        result = fullMatrix;
    else
        [~, score, ~] = pca(fullMatrix);
        result = score(:, 1 : dim);
    end    
end

function result = Standardize(matrix)
    means = mean(matrix, 1);
    stds = std(matrix, 0, 1);
    result = zeros(size(matrix));
    for i = 1 : size(matrix, 1)
        for j = 1 : size(matrix, 2)
            if stds(j) == 0
                result(i, j) = 1;
            else
                result(i, j) = (matrix(i, j) - means(j)) / stds(j);
            end
            
        end
    end
end

%***********************************
%              k-means             *
%***********************************/

function result = Distance(a, b)
    result = sum((a - b) .^ 2);
end

function [result, minDist] = IndexOfCluster(data, centers)
    n = size(centers, 1);
    minDist = -1;
    for i = 1 : n
        dist = Distance(data, centers{i});
        if minDist < 0 || dist < minDist
            minDist = dist;
            center = centers{i};
        end
    end
    for i = 1 : size(centers, 1)
        if center == centers{i}
            result = i;
            return
        end
    end
end

function result = TotalError(centers, clusters)
    if size(centers, 1) ~= size(clusters, 1)
        printError("Size dismatch");
    end
    result = 0;
    for i = 1 : size(centers, 1)
        center = centers{i};
        cluster = clusters{i};
        for j = 1 : size(cluster, 1)
            result = result + Distance(cluster{j}, center);
        end
    end
end

function result = Center(cluster)
    result = zeros(1, size(cluster{1}, 2));
    for i = 1 : size(cluster, 1)
        result = result + cluster{i};
    end
    result = result ./ size(cluster, 1);
end

function result = InitalCenters(all, k)
    n = size(all, 1);
    result = cell(k, 1);
    % first cluster
    %result{1} = Center(all); % alternative
    result{1} = all{randi(n)};
    % other clusters
    for i = 2 : k
        maxDist = -1;
        for j = 1 : size(all, 1)
            data = all{j};
            [~, dist] = IndexOfCluster(data, result(1 : i - 1));
            if maxDist < 0 || dist > maxDist
               maxDist = dist;
               farthest = data;
            end
        end
        result{i} = farthest;
    end
end

function result = MatchClusterIndices(all, clusters)
    n = size(all, 1);
    result = zeros(n, 1);
    for i = 1 : size(all, 1)
        flag = false;
        for cluster = 1 : size(clusters, 1)
            for j = 1 : size(clusters{cluster}, 1)                
                if all{i} == clusters{cluster}{j}
                    result(i) = cluster;
                    flag = true;
                    break;
                end
                if flag
                    break;
                end
            end
        end
    end
end

function [result, totalError] = KmeansOnce(matrix, k)
    n = size(matrix, 1);
    all = cell(n, 1);
    for i = 1 : n
        all{i} = matrix(i, :);
    end
    centers = InitalCenters(all, k);
    lastError = 0;
    while 1
        clusterSize = zeros(k, 1);
        clusters = cell(k, 1);
        for i = 1 : k
            clusters{i} = cell(n, 1);
        end
        for i = 1 : n
            index = IndexOfCluster(all{i}, centers);
            clusterSize(index) = clusterSize(index) + 1;
            clusters{index}{clusterSize(index)} = all{i};
        end
        for i = 1 : k
            clusters{i} = clusters{i}(1 : clusterSize(i));
        end
        centers = cell(k, 1);
        for i = 1 : k
            centers{i} = Center(clusters{i});
        end
        totalError = TotalError(centers, clusters);
        if abs(totalError - lastError) < 1e-3
            result = MatchClusterIndices(all, clusters);
            return
        end
        lastError = totalError;
    end
end

function result = Remapping(seg)
    mapping = zeros(max(max(seg)), 1);
    now = 1;
    for i = 1 : size(seg, 1)
        for j = 1 : size(seg, 2)
            if mapping(seg(i, j)) == 0
                mapping(seg(i, j)) = now;
                now = now + 1;
            end
        end
    end
    result = zeros(size(seg));
    for i = 1 : size(seg, 1)
        for j = 1 : size(seg, 2)
            result(i, j) = mapping(seg(i, j));
        end
    end
end

function [result, allTrails] = Kmeans(matrix, k, trail, mine) % k-means++
    MaxIters = 10000;
    minTotalDistance = -1;
    allTrails = cell(trail, 1);
    
    for i = 1 : trail
        if mine
            [idx, sumD] = KmeansOnce(matrix, k); % k-means inplemented by matlab is slow
        else
            [idx, ~, sumD] = kmeans(matrix, k, 'MaxIter',MaxIters); 
        end
        idx = Remapping(idx);
        % disp(idx');
        allTrails{i} = idx;
        totalDistance = sum(sumD);
        if minTotalDistance < 0 || totalDistance < minTotalDistance
            minTotalDistance = totalDistance;
            result = idx;
        end
    end
end

%***********************************
%                 a                *
%***********************************/
function fullMatrix = FullMatrix(textures, windowSize, flag15D)
    count = size(textures, 1);
    vectors = cell(count, 1);
    for i = 1 : count
       image = SubtractLocalMeans(textures{i}, windowSize);
       energies = ApplyLawsFilters(image);
       energies = PositiveEnergies(energies);
       matrix = AverageEnergy(energies);
       vector = ReshapeMatrixToVector(matrix);
       vector = CombineSymmetricPairs(vector, flag15D);
       vector = Normalize(vector);
       vectors{i} = vector;
    end
    fullMatrix = zeros(count, size(vectors{1}, 1));
    for i = 1 : count
        fullMatrix(i, :) = vectors{i};
    end
end

function [result, allResults] = TextureClassification(fullMatrix, flag3D)
    if flag3D
        reducedMatrix = ReduceDimension(fullMatrix, 3);
    else
        reducedMatrix = fullMatrix;
    end
    reducedMatrix = Standardize(reducedMatrix);    
    disp('Texture matrix');
    disp(reducedMatrix);
    global CLASSIFY_NUM;
    global MINE_KMEANS;
    [result, allResults] = Kmeans(reducedMatrix, CLASSIFY_NUM, 1000, MINE_KMEANS);
    if (flag3D)
        marks = 'ox^s';
        for i = 1 : CLASSIFY_NUM
            cluster = find(result == i);
            plot3(reducedMatrix(cluster, 1), reducedMatrix(cluster, 2), reducedMatrix(cluster, 3), marks(i));
            hold on;
        end
        grid on
        axis square
        hold off
    end
end

function result = CheckSegmentationCorrectRate(seg)
    standard = [1 2 3 4 1 4 1 2 3 2 3 4];
    result = 0;
    for i = 1 : 12
        if seg(i) == standard(i)
            result = result + 1;
        end
    end
    result = result / 12;
end

function problem1a
    global GRAY_CHANNEL;
    global IMAGE_FOLDER;
    global IMAGE_PREFIX;
    global IMAGE_SUFIX;
    global IMAGE_COUNT;
    global IMAGE_HEIGHT;
    global IMAGE_WIDTH;
    
    % load images
    textureImages = cell(IMAGE_COUNT, 1);
    for i = 1 : IMAGE_COUNT
       textureImages{i} = ReadRaw([IMAGE_FOLDER, '/', IMAGE_PREFIX, int2str(i), IMAGE_SUFIX], IMAGE_HEIGHT, IMAGE_WIDTH, GRAY_CHANNEL);
       %textureImages{i} = Shrink(textureImages{i});
    end
    
    % all arguments
    format short
    format compact
    tic;
    localWindowSize = 15;
    for convertTo15D = 0 : 1
        fullMatrix = FullMatrix(textureImages, localWindowSize, convertTo15D);
        if ~convertTo15D
            stds = std(fullMatrix, 0, 1);
            disp('Standard deviation of 25 feature dimensions (after normalization before standardization)');
            disp(stds);
        end
        for convertTo3D = 0 : 1
            disp(['Results for {CombiningSymmetricPairs:', int2str(convertTo15D), ', PCA3D:', int2str(convertTo3D), '}']);
            % calc            
            [result, allResults] = TextureClassification(fullMatrix, convertTo3D);
            if convertTo3D
                saveas(gcf, ['problem1_a_pca3d_', int2str(convertTo15D), '.fig']);
            end
            % disp all results
            count = 0;
            all = cell(size(allResults));
            for i = 1 : size(allResults, 1)
                notFound = true;
                for j = 1 : count
                    if all{j} == allResults{i}
                        notFound = false;
                        break;
                    end
                end
                if notFound
                    count = count + 1;
                    all{count} = allResults{i};
                end
            end
            all = all(1 : count)';
            all = cell2mat(all);
            all = all';
            correctRates = zeros(count, 1);
            for i = 1 : count
                correctRates(i, :) = CheckSegmentationCorrectRate(all(i, :));
            end
            temp = sortrows([correctRates, all], 1, 'descend');
            correctRates = temp(:, 1);
            all = temp(:, 2 : (12 + 1));
            disp(['All possible segmentation results (total ', int2str(count), '):']);
            disp(all);
            disp('Correct rates:');
            disp(correctRates);
            % disp min error result
            result = result';
            disp(['Segmentation result with minimal error (correct_rate ', num2str(CheckSegmentationCorrectRate(result)), '):']);
            disp(result);
        end
    end
    toc;
end

%***********************************
%                b & c             *
%***********************************/
function vectors = RawVectors(image, windowSize)
    % pre-processing subtract local mean
    image = SubtractLocalMeans(image, windowSize);
    % step 1
    energies = ApplyLawsFilters(image);
    energies = PositiveEnergies(energies);
    % step 2    
    matrices = AverageEnergies(energies, windowSize);
    % step 3
    vectors = ReshapeMatricesToVectors(matrices);
end

function vectors = FineVectors(vectors, flag15D)
    vectors = CombineAllSymmetricPairs(vectors, flag15D);
    vectors = NormalizeAll(vectors);
end

function result = TextureSegmentation(vectors, flagPCA)
    height = size(vectors, 1);
    width = size(vectors, 2);
    % setp 4
    vectors = reshape(vectors, height * width, 1);
    fullMatrix = zeros(height * width, size(vectors{1}, 1));
    for i = 1 : height * width
        fullMatrix(i, :) = vectors{i};
    end
    if flagPCA
        reducedMatrix = ReduceDimension(fullMatrix, flagPCA);
    else
        reducedMatrix = fullMatrix;
    end
    reducedMatrix = Standardize(reducedMatrix);    
    global SEGMENT_NUM;
    global MINE_KMEANS;
    seg = Kmeans(reducedMatrix, SEGMENT_NUM, 10, MINE_KMEANS); 
    seg = reshape(seg, height, width);
    global UINT8_MAX;
    result = uint8(round(Subtract(seg, ones(size(seg))) .* (UINT8_MAX / (SEGMENT_NUM - 1))));
end

function problem1b
    global GRAY_CHANNEL;
    global IMAGE_FOLDER;
    global COMB_FILENAME;
    global COMB_HEIGHT;
    global COMB_WIDTH;
    
    % load image
    image = ReadRaw([IMAGE_FOLDER, '/', COMB_FILENAME], COMB_HEIGHT, COMB_WIDTH, GRAY_CHANNEL);
    %img = Shrink(img);
    
    % all arguments
    tic;
    for localWindowSize = 15 : 8 : 55
        rawVectors = RawVectors(image, localWindowSize);
        for convertTo15D = 0 : 1
            vectors = FineVectors(rawVectors, convertTo15D);
            for Pca = 0 : 4 : 12
                result = TextureSegmentation(vectors, Pca);
                imshow(result);
                filename = ['problem1_b_output_', int2str(localWindowSize), '_', int2str(convertTo15D), '_', int2str(Pca)];
                saveas(gcf, [filename, '.fig']);
                WriteRaw(result, [filename, '.raw']);
            end
        end
    end
    toc;
end