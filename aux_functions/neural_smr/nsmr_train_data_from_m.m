function [S, T] = nsmr_train_data_from_m(m, xps)

% T_train is the target (label)
T = nsmr_reshape_4to2D(m);

f = @(x) dtd_smr_1d_fit2data_vec(x, xps);
S = nsmr_fit2data_matrix(T, f);