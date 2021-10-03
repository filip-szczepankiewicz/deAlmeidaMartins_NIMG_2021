function [S, T] = nsmr_ri_train_data_from_m(m, xps)

T = nsmr_reshape_4to2D(m);

f = @(x) smr_ri_fit2data_vec(x, xps);
S = nsmr_fit2data_matrix(T, f);