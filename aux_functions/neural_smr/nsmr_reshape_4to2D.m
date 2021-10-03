function out = nsmr_reshape_4to2D(in)
% out = nsmr_reshape_4to2D(in)
%
% 'in' - 4D array with size(in) = [sz_1 sz_2 sz_3 sz_4]
% 'out' - 2D array with size(out) = [sz_4 sz_1*sz_2*sz_3]

sz = size(in);
out = reshape(double(in), prod(sz(1:3)), sz(4));
out = permute(out, [2 1]);