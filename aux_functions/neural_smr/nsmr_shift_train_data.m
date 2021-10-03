function [T_shift, S_shift] = nsmr_shift_train_data( T_in, xps, shifts)

if (nargin < 3), shifts  = [.1*1e-9 5*1e-3]; end
% if (nargin < 3), n_shift = 50*1e3; end

% Define shifts
Diso_shift = shifts(1);
T2_shift   = shifts(2);

% Model bounds
maxp = sqrt(5 / (4*pi));
T_lb = [ 0.9*min(T_in(1,:))   0    0.07*1e-9   0.2*1e-9   -0.46   -maxp    -maxp    -maxp    -maxp     -maxp     .03     .03 ];
T_ub = [ 1.1*max(T_in(1,:))   1    1.33*1e-9    4*1e-9     0.86    maxp     maxp     maxp     maxp      maxp     .30      1  ];

sz = size(T_in);

% Choose n random entries from T_in
% ind     = 1 + round( rand( n_shift, 1 )*( sz(2) - 1) );
T_shift = T_in;

% Mean metrics
MD  = T_shift(2, :) .* T_shift(3, :) + (1 - T_shift(2, :)) .* T_shift(4, :);
MT2 = T_shift(2, :) .* T_shift(11, :) + (1 - T_shift(2, :)) .* T_shift(12, :);

% Shift Diso
T_shift(3, :) = T_shift(3, :) + Diso_shift;
T_shift(4, :) = (MD - T_shift(2, :) .* T_shift(3, :)) ./ (1 - T_shift(2, :));

% Shift T2
T_shift(11, :) = T_shift(11, :) + T2_shift;
T_shift(12, :) = (MT2 - T_shift(2, :) .* T_shift(11, :)) ./ (1 - T_shift(2, :));

% Catch & remove out-of-bound solutions
ind_out = T_shift > T_ub' | T_shift < T_lb';
ind_out = any(ind_out, 1);
T_shift(:, ind_out) = [];

f = @(x) dtd_smr_1d_fit2data_vec(x, xps);
S_shift = nsmr_fit2data_matrix(T_shift, f);
