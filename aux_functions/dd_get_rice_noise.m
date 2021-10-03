function S_rice = dd_get_rice_noise(S_in, snr, S_norm)
% function S_rice = dd_get_rice_noise(S_in, snr, S_norm)

if nargin < 3
    S_norm = 1;
end

sz = size(S_in);

if length(snr) == 1
    snr_v = snr * ones(1,sz(2));
elseif all(size(snr) == [1 sz(2)])
    snr_v = snr;
else
    disp('The ''snr'' variable must be either a scalar or an [1 x n_voxels] vector.')
    return
end

% Normalize signal 
S      = bsxfun(@rdivide, S_in, S_norm);  

% Add noise
S_real = S + bsxfun(@times, 1 ./ snr_v, randn(sz));
S_imag = bsxfun(@times, 1 ./ snr_v, randn(sz));
%
S_rice = bsxfun(@times, sqrt(S_real.^2 + S_imag.^2), S_norm);

