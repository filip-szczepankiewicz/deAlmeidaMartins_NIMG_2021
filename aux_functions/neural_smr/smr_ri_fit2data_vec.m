function c = smr_ri_fit2data_vec(m, xps)
% function c = smr_fit2data(m, xps)
%
% c = [c0(:); c2(:)], the rot-inv SH coefficients for l_max = 2
%
% c0 = k0
% c2 = p2*k2
%
% 

%%% Extract model parameters
%
s0      = m(1,:);
f       = m(2,:);
di_s    = m(3,:);
dd_s    = ones(size(s0));
di_z    = m(4,:);
dd_z    = m(5,:);
t2_s    = m(6,:);
t2_z    = m(7,:);
p0      = ones(size(s0));
p2      = m(8,:);


%%% Diffusion coefficients
%
a_s = -xps.b * di_s .* (1 - xps.b_delta * dd_s);
a_z = -xps.b * di_z .* (1 - xps.b_delta * dd_z);
%
A_s = 3 * (xps.b * di_s) .* (xps.b_delta * dd_s);
A_z = 3 * (xps.b * di_z) .* (xps.b_delta * dd_z);
%
[i0_s, i2_s] = dtd_smr_i0i2(A_s); % = INT[exp(-A*x^2) * Y_l], 0 to 1 (integrates over Y_l, manus definition is over L_l)
[i0_z, i2_z] = dtd_smr_i0i2(A_z); 
%
k0_diff_s =  exp(a_s) .* i0_s;
k0_diff_z =  exp(a_z) .* i0_z;
%
k2_diff_s =  exp(a_s) .* i2_s;
k2_diff_z =  exp(a_z) .* i2_z;


%%% Relaxation attenuation
%
At2s = exp(-xps.te * ( 1 ./ t2_s ));
At2z = exp(-xps.te * ( 1 ./ t2_z ));

%%% Total kernel
%
%%% Signal
k0 = s0 .* (f .* At2s .* k0_diff_s + (1 - f) .* At2z .* k0_diff_z);% / sqrt(4 * pi);
k2 = s0 .* (f .* At2s .* k2_diff_s + (1 - f) .* At2z .* k2_diff_z);% / sqrt(20 * pi);

%%% Predicted RotInv coefficients
%
c0 = abs(k0) .* p0;
c2 = abs(k2) .* p2;
c = cat(1, c0, c2);

end