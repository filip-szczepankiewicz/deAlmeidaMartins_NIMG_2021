function [axh, ph] = mplot_corr(x, y, col, axh)
% function [axh, ph] = mplot_corr(x, y, col, axh)

if (nargin < 3), col = bsxfun(@times, ones(numel(x),1), [1 0 0]); end
if (nargin < 4), axh = gca; end

n_samp    = min([numel(x) 24*1e3]);
rand_indx = randperm(numel(x)); 
rand_indx = rand_indx(1:n_samp);

ph = scatter(x(rand_indx), y(rand_indx), 4, col(rand_indx, :), ...
        'o', 'SizeData', 4, 'Linewidth', .5);
% ph = scatter(x(rand_indx), y(rand_indx), 4, col(rand_indx, :), ...
%         'filled', 'o', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', .05, 'SizeData', 4);