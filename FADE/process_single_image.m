% Process single image and return FADE density score
% Usage: octave-cli process_single_image.m <image_path>

pkg load image
pkg load statistics

% Get image path from command line argument
args = argv();
if length(args) < 1
    fprintf('Error: No image path provided\n');
    fprintf('Usage: octave-cli process_single_image.m <image_path>\n');
    exit(1);
end

image_path = args{1};

% Check if file exists
if ~exist(image_path, 'file')
    fprintf('Error: Image file not found: %s\n', image_path);
    exit(1);
end

try
    % Read image and calculate FADE density
    image = imread(image_path);
    density = FADE(image);
    
    % Output only the density score (for easy parsing)
    fprintf('%.6f\n', density);
    
catch err
    fprintf('Error processing image: %s\n', err.message);
    exit(1);
end
