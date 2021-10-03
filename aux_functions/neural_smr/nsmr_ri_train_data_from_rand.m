function [S, T, T_nam, T_ub, T_lb] = nsmr_ri_train_data_from_rand(xps, n, T_lims)
% function [S, T, T_nam, T_ub, T_lb] = nsmr_ri_train_data_from_rand(xps, n, T_lims)
%%% model parameters
%             1       2       3        4         5        6        7      8
T_nam   =  { 's0'    'fs'   'di_s'   'di_z'    'dd_z'   't2_s'   't2_z'   'p2'};

if (nargin < 3)
    T_lims = [ .5      0    0.07*1e-9   0.2*1e-9   -0.46     .03     .03    0; ...
                2      1    1.33*1e-9    4*1e-9     0.86     .30      1     1]; 
end

T_lb  =  T_lims(1,:);
T_ub  =  T_lims(2,:);

T = T_lb' + bsxfun(@times,T_ub' - T_lb',rand(length(T_ub),n));

f = @(x) smr_ri_fit2data_vec(x, xps);
S = nsmr_fit2data_matrix(T, f);

