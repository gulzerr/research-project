% FADE example

clear all; close all; clc; 
pkg load image  % Load image processing package
pkg load statistics  % Load statistics package for nanvar

% Process images and collect results
image_ids = {'test_image1.png', 'test_image2.JPG', 'test_image3.jpg'};
scores = [];

for i = 1:length(image_ids)
    image = imread(image_ids{i}); 
    density = FADE(image);
    scores(i) = density;
    fprintf('Image: %s, Density Score: %.4f\n', image_ids{i}, density);
end

% Save results to CSV for parquet conversion
fid = fopen('density_results.csv', 'w');
fprintf(fid, 'image_id,score\n');
for i = 1:length(image_ids)
    fprintf(fid, '%s,%.6f\n', image_ids{i}, scores(i));
end
fclose(fid);

fprintf('\nResults saved to density_results.csv\n');
fprintf('Run: python convert_to_parquet.py to convert to parquet format\n');

