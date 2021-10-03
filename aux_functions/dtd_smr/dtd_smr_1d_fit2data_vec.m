function s = dtd_smr_1d_fit2data_vec(m, xps)
% function s = smr_fit2data(m, xps)
%
% s = s0 * [ f * Ads * At2s + (1 - f) * Adz * At2z]
%

% Convert gradient vectors from Cartesian to Spherical
x       = xps.u(:,1);
y       = xps.u(:,2);
z       = xps.u(:,3);
[phi, theta] = cart2sph(x, y, z);
phi     = phi + pi;       % [-pi pi]  --> [0 2pi], longitude
theta   = pi / 2 - theta; % latitude  --> co-latitude

%%% Diffusion attenuation
%
%
a_s = -xps.b * m(3,:) .* (1 - xps.b_delta * ones(size(m(1,:))));
a_z = -xps.b * m(4,:) .* (1 - xps.b_delta * m(5,:));
%
A_s = 3 * (xps.b * m(3,:)) .* (xps.b_delta * ones(size(m(1,:))));
A_z = 3 * (xps.b * m(4,:)) .* (xps.b_delta * m(5,:));
%
[i0_s, i2_s] = dtd_smr_i0i2(A_s); % = INT[exp(-A*x^2) * Y_l], 0 to 1 (integrates over Y_l, manus definition is over L_l)
[i0_z, i2_z] = dtd_smr_i0i2(A_z); 
%
% Orientational information
y20 = dtd_smr_spha(2,  0, theta, phi);
y21 = dtd_smr_spha(2,  1, theta, phi);
y22 = dtd_smr_spha(2,  2, theta, phi);
p2m_y2m_sum    = ...
    y20 * m(6,:) + ...
    real(y21) * m(7,:) * ( 2) + ...
    imag(y21) * m(8,:) * (-2) + ...
    real(y22) * m(9,:) * ( 2) + ...
    imag(y22) * m(10,:) * (-2); 

y00 = 1 / sqrt(4 * pi); % By definition
p00 = 1 / sqrt(4 * pi); % ODF normalization
%
Ads     = exp(a_s) .* (i0_s * p00 * y00 * sqrt(4*pi) + i2_s .* p2m_y2m_sum * sqrt(4*pi/5));
Adz     = exp(a_z) .* (i0_z * p00 * y00 * sqrt(4*pi) + i2_z .* p2m_y2m_sum * sqrt(4*pi/5));

%%% Relaxation attenuation
%
%
At2s = exp(-xps.te * ( 1 ./ m(11,:) ));
At2z = exp(-xps.te * ( 1 ./ m(12,:) ));

%%% Signal
s = m(1,:) .* ( m(2,:) .* Ads .* At2s + (1 - m(2,:)) .* Adz .* At2z );


end