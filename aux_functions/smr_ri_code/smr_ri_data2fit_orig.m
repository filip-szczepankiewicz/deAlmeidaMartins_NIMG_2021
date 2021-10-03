function m = smr_ri_data2fit(signal, xps, opt)
% function m = smr_ri_data2fit(signal, xps, opt)
%
%
% s0      = m(1);
% f       = m(2);
% di_s    = m(3);
% di_z    = m(4);
% dd_z    = m(5);
% t2_s    = m(6);
% t2_z    = m(7);
% p2      = m(8);    

if (nargin < 3), opt = smr_opt(); end


%%% RI-projection
%
[c0, c2] = smr_project_ri(signal, xps);
c = [c0(:); c2(:)];

%%% Experiment
%
n_dir   = histcounts(xps.s_ind, max(xps.s_ind));
w       = [n_dir(:); n_dir(:) / 5];  % Novikov 2018
%
opt_pa  = mdm_opt;
opt_pa.do_use_s_ind = 1;
xps     = mdm_xps_pa(xps, opt_pa);

%%% Normalization
%
ms = max(signal);
%               1     2       3       4       5       6     7       8
%               s0    fs      di_s    di_z    dd_z    t2_s  t2_z    p2
unit_to_SI =   [ms    1       1e-9    1e-9    1       1e-3  1e-3    1];
%

%%% Set bounds
%           1       2       3       4       5       6       7       8
%           s0      fs      di_s    di_z    dd_z    t2_s    t2_z    p2
t_lb      = [0      eps     0.07    0.2    -0.4634  30      30      0];
t_ub      = [40     1-eps   1.33    4       0.8636  300     1e3     1];

% Handle fixing (for iteration)
if (isfield(opt, 'f_min') && isfield(opt, 'f_max'))
    t_lb(2) = opt.f_min;
    t_ub(2) = opt.f_max;
end

lambda = mean(signal);

%%% Fitting function
    function s = my_1d_fit2data(t,varargin)
        
        % m2t
        m = t .* unit_to_SI;
        
        % coefficient prediction
        c_fit = smr_ri_fit2data(m, xps);
        
        
        % Penalties for keeping both da/r_z within [0.2 4]
        %
        p_factor = 1e1;
        %
        da_z = t(4) * (1 + 2 * t(5));
        dr_z = t(4) * (1 - 1 * t(5));
        %
        penalty_ah =  exp(max(0,  p_factor * (da_z - t_ub(4)))) - 1;
        penalty_rh =  exp(max(0,  p_factor * (dr_z - t_ub(4)))) - 1;
        penalty_al =  exp(max(0,  p_factor * (t_lb(4) - da_z))) - 1;
        penalty_rl =  exp(max(0,  p_factor * (t_lb(4) - dr_z))) - 1;
        %
        penalty = sum([penalty_ah penalty_rh penalty_al penalty_rl]);
        
        % Compose weighted output
        s = [sqrt(w) .* c_fit; sum(penalty) * lambda];
    end
% tic
% Perform fitting iteratively
r = Inf;
for c_rep = 1:opt.n_rep
    
    % Possibly provide an initial guess
    if (isempty(opt.init_guess))
        g   = t_lb + rand(size(t_lb)) .* (t_ub - t_lb); % guess
    else
        g = opt.init_guess;
    end
    %
    warning off;
    [t_tmp, r_tmp] = lsqcurvefit(@(t,varargin) my_1d_fit2data(t), g, [], ...
        [sqrt(w) .* c; 0], t_lb, t_ub, opt.lsq_opts);
    warning on;
    
    if (r_tmp < r)
        r = r_tmp;
        t = t_tmp;
    end
end
% toc
m = t .* unit_to_SI;

end