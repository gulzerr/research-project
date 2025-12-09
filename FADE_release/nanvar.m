function v = nanvar(x, varargin)
    % Simple nanvar implementation - variance ignoring NaN values
    % Handles matrices by computing variance along columns (like MATLAB)
    if isvector(x)
        x = x(~isnan(x));
        v = var(x, varargin{:});
    else
        % For matrices, compute variance for each column
        v = zeros(1, size(x, 2));
        for i = 1:size(x, 2)
            col = x(:, i);
            col = col(~isnan(col));
            if isempty(col)
                v(i) = NaN;
            else
                if nargin > 1
                    v(i) = var(col, varargin{1});
                else
                    v(i) = var(col);
                end
            end
        end
    end
end
