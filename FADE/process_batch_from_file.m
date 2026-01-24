function process_batch_from_file(list_file, out_file, gpu_index)
% process_batch_from_file(list_file, out_file, gpu_index)
% Reads image paths from a text file, calls process_single_image for each image,
% and writes a Parquet table with columns: image_path,score. Falls back to CSV
% if parquetwrite is unavailable.
%
% If gpu_index is provided (numeric), attempts to select that GPU: gpuDevice(gpu_index).

if nargin >= 3 && ~isempty(gpu_index)
    try
        gpuDevice(double(gpu_index));
        fprintf('Selected GPU %d\n', int32(gpu_index));
    catch ME
        fprintf('Warning selecting GPU: %s\n', getReport(ME));
    end
end

fid = fopen(list_file, 'r');
if fid == -1
    error('Could not open list file: %s', list_file);
end
paths = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
paths = paths{1};

n = numel(paths);
image_paths = cell(n,1);
scores = nan(n,1);

for i = 1:n
    p = paths{i};
    image_paths{i} = p;
    try
        % Attempt to call existing process_single_image; capture its stdout
        out = evalc(sprintf("process_single_image('%s')", strrep(p,'''','''''')));
        val = str2double(strtrim(out));
        if ~isnan(val)
            scores(i) = val;
        else
            scores(i) = NaN;
        end
    catch ME
        fprintf('Error processing %s: %s\n', p, getReport(ME));
        scores(i) = NaN;
    end
end

T = table(image_paths, scores, 'VariableNames', {'image_path','score'});
% Attempt to write Parquet; fall back to CSV if parquetwrite is unavailable
try
    parquetwrite(char(out_file), T);
    fprintf('Wrote %d results to %s\n', n, out_file);
catch ME
    try
        writetable(T, char(out_file));
        fprintf('Wrote %d results (fallback CSV) to %s\n', n, out_file);
    catch ME2
        fprintf('Failed to write output: %s\n', getReport(ME2));
    end
end
end
