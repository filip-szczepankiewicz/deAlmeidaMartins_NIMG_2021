function [S, T, T_nam, T_ub, T_lb] = nsmr_train_data_from_rand(xps, n, T_lims)
% function [S, T, T_nam, T_ub, T_lb] = nsmr_train_data_from_rand(xps, n, T_lims)
%%% model parameters
%             1       2       3           4         5        6        7        8        9         10      11       12
T_nam   =  { 's0'    'fs'   'di_s'     'di_z'    'dd_z'    'p20'    'p21r'   'p21i'   'p22r'    'p22i'   't2_s'  't2_z'};

if (nargin < 3)
    maxp = sqrt(5 / (4*pi));
    T_lims = [ .5      0    0.07*1e-9   0.2*1e-9   -0.46     -maxp    -maxp    -maxp    -maxp     -maxp     .03     .03; ...
                2      1    1.33*1e-9    4*1e-9     0.86      maxp     maxp     maxp     maxp      maxp     .30      1 ]; 
end

T_lb  =  T_lims(1,:);
T_ub  =  T_lims(2,:);

T = T_lb' + bsxfun(@times,T_ub' - T_lb',rand(length(T_ub),n));

% generate SphHarm coeffcients using a rand distribution of points on a 5-D sphere
P = randpoints_nsphere( n, 5, T_ub(6));
T(6:10,:) = P';

f = @(x) dtd_smr_1d_fit2data_vec(x, xps);
S = nsmr_fit2data_matrix(T, f);