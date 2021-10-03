function [S_norm, S_norm_cnst] = nsmr_normalize_signal(S, indx, method)
% function S_norm = nsmr_normalize_signal(S, indx, method)
% Perform voxel-wise signal normalization using the normalization constant calculated
% from the 'method' = {median, mean, max} of the data points selected by 'indx' 

if (nargin < 3), method = 'median'; end

switch method
    
    case 'median'
        S_norm_cnst = median(S(indx, :), 1);
                
    case 'mean'
        S_norm_cnst = mean(S(indx, :), 1);
                
    case 'max'
        S_norm_cnst = max(S(indx, :), 1);
                        
end

S_norm = bsxfun(@rdivide, S, S_norm_cnst);