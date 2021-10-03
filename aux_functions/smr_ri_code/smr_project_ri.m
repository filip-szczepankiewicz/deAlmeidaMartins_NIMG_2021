function [c0, c2, sp_slm] = smr_project_ri(s, xps, do_plot)

if (nargin < 3), do_plot = 0; end


%%% Parse input
%
s               = s(:);
shell_ind_list  = unique(xps.s_ind);
shell_n         = numel(shell_ind_list);

%%% Obtain coefficients (shell-wise)
%
slm     = zeros([shell_n  6]);
sp_slm  = zeros(size(s));
for c_shell = 1:shell_n
    
    ind     = xps.s_ind == shell_ind_list(c_shell);
    n_dir   = sum(ind);
    
    % Extract directions (convert cartesian to spherical)
    %
    x       = xps.u(ind,1);
    y       = xps.u(ind,2);
    z       = xps.u(ind,3);
    [phi, theta] = cart2sph(x, y, z);
    phi     = phi + pi;       % [-pi pi]  --> [0 2pi]
    theta   = pi / 2 - theta; % elevation --> inclination
    
    % SH bases for these directions
    %
    y00  = dtd_smr_spha(0,   0, theta, phi);
    y20  = dtd_smr_spha(2,   0, theta, phi);
    y2p1 = dtd_smr_spha(2,  +1, theta, phi);
    y2p2 = dtd_smr_spha(2,  +2, theta, phi);
    y2m1 = (-1)^1 * conj(y2p1);
    y2m2 = (-1)^2 * conj(y2p2);    
    
    
    %%% Project onto the sh bases by solving the linear system:
    % Y    = XB, where
    % Yi   = S(theta_i,phi_i)
    % Xi   = [y00_i y2m2_i y2m1_i y2p0 y2p1_i y2p2_i]
    % B    = [S_00  S_2-2  S_2-1  S_20 S_21   S_22]'
    %    
    Y           = s(ind);
    X           = [y00 y2m2 y2m1 y20 y2p1 y2p2];   
    B           = pinv(X) * Y;        
    %    
    B([1 4])        = real(B([1 4])); % Only m~=0 can be complex
    %
    slm(c_shell,:)  = B; 
    sp_slm(ind)     = real(X*B);    
end

% Extract s0 and convert s2m into rot-inv s2
%
% c0 = sqrt( sum( abs(slm(:, 1) / sqrt(4 * pi) ).^2, 2) ) / sqrt(1 / (4*pi));
% N  = sqrt(5 / (4 * pi));
% c2 = sqrt( sum( abs(slm(:, 2:6) / sqrt(5 / (4 * pi))).^2, 2) ) / N;
c0 = sqrt( sum( abs(slm(:, 1)).^2, 2) );
c2 = sqrt( sum( abs(slm(:, 2:6) ).^2, 2) );

%%% Plotting
%
if (do_plot)
    figure(636)
    set(gcf, 'color', 'w')
    %
    plot(s, '.k');
    hold on
    plot(sp_slm, 'or');
    title(['SS = ' num2str(sum((s - sp_slm).^2))])
    set(gca, 'box', 'off');
    axis square
end

end