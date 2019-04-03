%***********************************
%  Name: Zongjian Li               *
%  USC ID: 6503378943              *
%  USC Email: zongjian@usc.edu     *
%  Submission Date: 22th,Jan 2019  *
%***********************************/

function main
    % clear all
    % close all    
    c2('../../HW1_images');
end

% This function uses BM3D function which can be found at http://www.cs.tut.fi/~foi/GCF-BM3D/
function [PSNR_Biased, PSNR_Uniased] = c2(filepath)
    randn('seed', 0);
    MAX = 255;
    row = 256;
    col = 256;
    sigma = 1;
    noiseFilename = [filepath, '/', 'pepper_dark_noise.raw']; 
    originalFilename = [filepath, '/', 'pepper_dark.raw'];

    file = fopen(noiseFilename, 'r');
    raw = fread(file, row * col, 'uint8=>uint8'); 
    noise = reshape(raw, row, col);
    noise = noise';
    noise = im2double(noise); 

    transNoise = zeros(row, col);
    for i = 1: row
        for j = 1: col
            transNoise(i, j) = 2 * sqrt(noise(i,j) * MAX + 3.0 / 8) / MAX;
        end
    end

    [~, filtered] = BM3D(1, transNoise, sigma); % BM3D executes here

    transFilteredBiased = zeros(row, col);
    transFilteredUnbiased = zeros(row, col);
    for i = 1: row
        for j = 1: col
            transFilteredBiased(i, j) = ((filtered(i, j) * MAX / 2.0) ^ 2 - 3.0 / 8) / MAX;
            transFilteredUnbiased(i, j) = ((filtered(i, j) * MAX / 2.0) ^ 2 - 1.0 / 8) / MAX;
        end
    end

    file = fopen(originalFilename, 'r');
    raw = fread(file, row * col, 'uint8=>uint8'); 
    original = reshape(raw, row, col);
    original = original';
    original = im2double(original);

    PSNR_Biased = 10 * log10(1 / mean((original(:) - transFilteredBiased(:)) .^ 2)) % code from BM3d.m line 41, my PSNR function is written in C++!
    PSNR_Uniased = 10 * log10(1 / mean((original(:) - transFilteredUnbiased(:)) .^ 2)) % code from BM3d.m line 41, my PSNR function is written in C++!

    transFilteredBiased = mat2gray(transFilteredBiased, [0, 1]);
    transFilteredUnbiased = mat2gray(transFilteredUnbiased, [0, 1]);
    figure; imshow(noise);   
    figure; imshow(transFilteredBiased);
    figure; imshow(transFilteredUnbiased);
end

