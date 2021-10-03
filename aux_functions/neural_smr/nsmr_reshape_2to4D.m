function out = nsmr_reshape_2to4D(in, sz)
% out = nsmr_reshape_4to2D(in)
%
% 'in' - 2D array
% 'sz' - 3D size array where prod(sz(1:3)) = size(in, 2)  
% 'out' - 4D array with size(out) = [sz(1) sz(2) sz(3) size(in, 1)]

sz  = [ sz size(in, 1) ];
out = permute(in, [2 1]);
out = reshape( out, sz);