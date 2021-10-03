function c = smr_ri_fit2data(m, xps)
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
s0      = m(1);
f       = m(2);
di_s    = m(3);
dd_s    = 1;
di_z    = m(4);
dd_z    = m(5);
t2_s    = m(6);
t2_z    = m(7);
p0      = 1;
p2      = m(8);



%%% Diffusion coefficients
%
a_s = -xps.b * di_s .* (1 - xps.b_delta * dd_s);
a_z = -xps.b * di_z .* (1 - xps.b_delta * dd_z);
%
A_s = 3 * xps.b .* xps.b_delta * di_s * dd_s;
A_z = 3 * xps.b .* xps.b_delta * di_z * dd_z;
%
[C0_s, C2_s] = smr_c0c2(A_s);
[C0_z, C2_z] = smr_c0c2(A_z);
%
k0_diff_s =  1/2 * exp(a_s) .* C0_s;
k0_diff_z =  1/2 * exp(a_z) .* C0_z;
%
k2_diff_s =  1/2 * exp(a_s) .* C2_s;
k2_diff_z =  1/2 * exp(a_z) .* C2_z;


%%% Relaxation coefficients
%
a_relax_s = exp(-xps.te/t2_s);
a_relax_z = exp(-xps.te/t2_z);


%%% Total kernel
%
k0 = s0 * (f * a_relax_s .* k0_diff_s + (1 - f) * a_relax_z .* k0_diff_z);
k2 = s0 * (f * a_relax_s .* k2_diff_s + (1 - f) * a_relax_z .* k2_diff_z);

%%% Predicted RotInv coefficients
%
c0 = abs(k0) * p0;
c2 = abs(k2) * p2;
c = [c0(:); c2(:)];

end