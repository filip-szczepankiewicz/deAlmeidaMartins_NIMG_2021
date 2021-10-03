function [T_mut, S_mut] = nsmr_mutate_train_data( T_in, xps, n_mut, fuzz)
% function [T_mut, S_mut] = nsmr_mutate_train_data( T_in, xps, n_mut, fuzz)

if (nargin < 4), fuzz = .2; end
if (nargin < 3), n_mut = 50*1e3; end

% Model bounds
maxp = sqrt(5 / (4*pi));
T_lb = [ 0.9*min(T_in(1,:))   0    0.07*1e-9   0.2*1e-9   -0.46   -maxp    -maxp    -maxp    -maxp     -maxp     .03     .03 ];
T_ub = [ 1.1*max(T_in(1,:))   1    1.33*1e-9    4*1e-9     0.86    maxp     maxp     maxp     maxp      maxp     .30      1  ];

sz = size(T_in);

% Choose n random entries from T_in
ind = 1 + round( rand( n_mut, 1 )*( sz(2) - 1) );
T_mut = T_in(:,ind);

% mutate the n chosen entries
T_mut = T_mut .* (1 + fuzz * randn( sz(1), n_mut ));

frac_out = 1;
while frac_out > .05
    % Catch out-of-bound solutions
    ind_out = T_mut > T_ub' | T_mut < T_lb';
    ind_out = any(ind_out, 1);
    frac_out = sum(ind_out) / n_mut;
        
    % Redo mutations for out-of-bound solutions
    T_mut(:, ind_out) = T_in(:, ind(ind_out)) .* ( 1 + fuzz * randn(sz(1), sum(ind_out)) );
end
% Remove out-of-bound solutions
T_mut(:, ind_out) = [];

f = @(x) dtd_smr_1d_fit2data_vec(x, xps);
S_mut = nsmr_fit2data_matrix(T_mut, f);
