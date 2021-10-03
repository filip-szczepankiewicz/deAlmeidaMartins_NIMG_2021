clear; close all;

do_mask    = 1; % Mask skull
do_WM_mask = 0; % Focus on WM

net_name   = 'net_ri_optimized_182183'; % Select trained network 
invivo_dir = '113_FysikDiffusion_185'; % In Vivo dataset to deploy the NN
%
dir_path   = pwd;
net_fn     = fullfile(dir_path, 'Train_Networks', [net_name '.mat']);

% Load previously trained network
load(net_fn)
%
if contains(net_name, 'deep')
    xps    = userdata.xps;
    T_ind  = userdata.T_ind;
    T_name = userdata.dataset_pars.T_name;
    %
    feat_norm_pars = userdata.feat_norm_pars;
    norm_function  = userdata.dataset_pars.s_norm_pars.function;
    inputs         = userdata.dataset_pars.s_norm_pars.function_input;
    %
    snr_max = max(userdata.dataset_pars.snr); 
    snr_min = min(userdata.dataset_pars.snr);
else
    net    = nets{end}; clear nets
    xps    = net.userdata.xps;
    T_ind  = net.userdata.T_ind;
    T_name = net.userdata.dataset_pars.T_name;
    %
    feat_norm_pars = net.userdata.feat_norm_pars;
    norm_function  = net.userdata.dataset_pars.s_norm_pars.function;
    inputs         = net.userdata.dataset_pars.s_norm_pars.function_input;
    %
    snr_max = max(net.userdata.dataset_pars.snr);
    snr_min = min(net.userdata.dataset_pars.snr);
end

%% Test on uniformly sampled signals

n_unif  = 100*1e3;

% Generate random synthetic dataset
maxp   = sqrt(5 / (4*pi));
T_lims = [ 0.5   0    0.07*1e-9   0.2*1e-9   -0.46   -maxp    -maxp    -maxp    -maxp     -maxp     .03     .03; ...
    4    1    1.33*1e-9    4*1e-9     0.86    maxp     maxp     maxp     maxp      maxp     .30      1 ];
%
[S_unif, T_unif, ~, ~, ~] = nsmr_train_data_from_rand(xps, n_unif, T_lims);

% Add noise
snr    = unifrnd(snr_min, snr_max, 1, size(S_unif, 2));
S_unif = dd_get_rice_noise(S_unif, snr, T_unif(1, :));

% Project into SH basis
S0_unif = zeros(max(xps.s_ind), size(S_unif, 2)); S2_unif = S0_unif;
%
parfor n_vxl = 1:size(S_unif, 2)
    [S0_unif(:, n_vxl), S2_unif(:, n_vxl), ~] =...
        smr_project_ri(S_unif(:,n_vxl), xps, 0);
end

% Normalize signal
eval(['[S0_unif, S0_norm_cnst] = ' norm_function_s0 '( S0_unif, inputs_s0{1}, inputs_s0{2});'])
S2_unif = bsxfun( @rdivide, S2_unif, S0_norm_cnst);
%
[S0_unif, ~] = removeconstantrows(S0_unif); % Remove constant rows

% RotInv parameter targets
m_ri = nsmr_reshape_2to4D(T_unif, [size(T_unif, 2) 1 1]);
m_ri = nsmr_m2rotinvm(m_ri);
m_ri = msf_notfinite2zero(m_ri);
%
T_unif = nsmr_reshape_4to2D(m_ri);
T_unif       = T_unif(T_ind, :);

if only_s0
    S_unif = S0_unif;
else
    S_unif = cat(1, S0_unif, S2_unif);
end

% Predict data
if contains(net_name, 'deep')
    P_unif = predict(net, S_unif');
    P_unif = P_unif';
else
    P_unif = net(S_unif);
end

% Convert predictions to SI units
[P_unif, ~] = nsmr_feature_undo_norm(P_unif, feat_norm_pars);


%% Test on in vivo data

% Connect to in vivo data
data_dir = fullfile('D:\Users\joao\Data\Bjorns_Paper', invivo_dir);
nii_fn   = fullfile(data_dir, 'output', 'dmri_mc_topup_gibbs.nii.gz');

% Load NLLS fit results
mfs_fn  = fullfile(data_dir, 'NII_RES', 'dtd_smr', 'mfs.mat');
mfs     = mdm_mfs_load(mfs_fn); % load mfs structure
m       = mfs.m(:,:,:,1:(end-1)); % remove fit residuals
m_ri    = nsmr_m2rotinvm(m); % convert to RotInv parameters
%
T_brain = nsmr_reshape_4to2D(m_ri);

% Generate synthetic dataset from experimental m structure
[S_brain, ~] = nsmr_train_data_from_m(m, xps);

% Add noise
snr     = unifrnd(snr_min, snr_max, 1, size(S_brain, 2));
S_brain = dd_get_rice_noise(S_brain, snr, T_brain(1, :));

% Project into SH basis
S0_brain = zeros(max(xps.s_ind), size(S_brain, 2)); S2_brain = S0_brain;
%
parfor n_vxl = 1:size(S_brain, 2)
    [S0_brain(:, n_vxl), S2_brain(:, n_vxl), ~] =...
        smr_project_ri(S_brain(:,n_vxl), xps, 0);
end

% Normalize signal
eval(['[S0_brain, S0_norm_cnst] = ' norm_function_s0 '( S0_brain, inputs_s0{1}, inputs_s0{2});'])
S2_brain = bsxfun( @rdivide, S2_brain, S0_norm_cnst);
%
[S0_brain, ~] = removeconstantrows(S0_brain); % Remove constant rows
T_brain       = T_brain(T_ind, :);

if only_s0
    S_brain = S0_brain;
else
    S_brain = cat(1, S0_brain, S2_brain);
end

% Predict data
if contains(net_name, 'deep')
    P_brain = predict(net, S_brain');
    P_brain = P_brain';
else
    P_brain = net(S_brain);
end
%
% Convert predictions to SI units
[P_brain, ~] = nsmr_feature_undo_norm( P_brain, feat_norm_pars);


% Load experimental data
[I, ~] = mdm_nii_read(nii_fn); % Load data
S_exp  = nsmr_reshape_4to2D( double(I) ); clear I

% Project into SH basis
S0_exp = zeros(max(xps_pa.s_ind), size(S_exp, 2)); S2_exp = S0_exp;
tic
parfor i = 1:size(S_exp,2)
    [S0_exp(:, i), S2_exp(:, i), ~] = smr_project_ri(S_exp(:,i), xps, 0);
end
toc

% Normalize signal
eval(['[S0_exp, S0_norm_cnst] = ' norm_function_s0 '( S0_exp, inputs_s0{1}, inputs_s0{2});'])
S2_exp = bsxfun( @rdivide, S2_exp, S0_norm_cnst);
[S0_exp, ~] = removeconstantrows(S0_exp); % Remove constant rows
%
if only_s0
    S_exp = S0_exp;
else
    S_exp = cat(1, S0_exp, S2_exp);
end

% Predict data
if contains(net_name, 'deep')
    P_exp = predict(net, S_exp');
    P_exp = P_exp';
else
    P_exp = net(S_exp);
end
%
% Convert predictions to SI units
[P_exp, ~] = nsmr_feature_undo_norm( P_exp, feat_norm_pars);


%% Masking

% Mask out non-brain areas (skull_mask) and non-WM areas (mask)
if do_mask
    uFA_nii  = fullfile(data_dir, 'NII_RES', 'maps', 'dtd_covariance_uFA.nii.gz');
    mask     = nsmr_make_mask(nii_fn, mfs_fn, do_WM_mask, uFA_nii);
    mask_bol = logical( mask(:) );
else
    mask     = ones( size(m_ri, 1:3) );
    mask_bol = logical( mask(:) );
end

%% Plot maps

m_targ = m_ri;
%
m_pred = nsmr_reshape_2to4D(P_brain, size(m_targ, 1:3));
m_pred = cat(4, ones(size(m_targ, 1:3)), m_pred);
m_pred = nsmr_m2rotinvm(m_pred);
m_pred = msf_notfinite2zero(m_pred);
%
m_exp  = nsmr_reshape_2to4D(P_exp, size(m_targ, 1:3));
m_exp  = cat(4, ones(size(m_targ, 1:3)), m_exp);
m_exp  = nsmr_m2rotinvm(m_exp);
m_exp  = msf_notfinite2zero(m_exp);

slice = 19;
d     = 3;

m_v    = [ 2       3          4         5      6      7       8];
m_lims = [ 0    0.0*1e-9   0.0*1e-9   -0.5    0.0    0.0     0.0 ; ...
           1    2.0*1e-9   2.0*1e-9    1.0    .20    .20     1.0];

as.nr     = numel(m_v);
as.nc     = 4;
as.l_marg = 0;
as.r_marg = 0;
as.u_marg = 0;
as.b_marg = 0;
as.blk_sp = -.01;
style     = 'cols';
papersize = [10 10];
ext       = '-dmeta';
res       = 'Inf';

figure(1), clf

% Maps Standard
for i = 1:numel(m_v)
    
    % Target
    clear im3d
    im3d = (m_targ(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 4*(i-1) + 1, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    % Pred
    clear im3d
    im3d = (m_pred(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 4*(i-1) + 2, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    % Diff
    clear im3d
    im3d = ( m_pred(:, :, :, m_v(i)) - m_targ(:, :, :, m_v(i)) );
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 4*(i-1) + 3, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, mplot_cmaphotcold(30))
    set(axh, 'CLim', [-1 1] * .5 * m_lims(2, i))   
    axis(axh,'off')
    
    % Target
    clear im3d
    im3d = (m_exp(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 4*(i-1) + 4, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')        
    %
    
end

fig_fn = mkdir(fullfile(dir_path, 'Figures', 'Maps_standard'));
fig_fn = fullfile(fig_fn, 'net_maps');
fig    = jm_save_fig(papersize, fig_fn, ext, res);

%% Plot Correlations

T_unif_p2   = nsmr_T_p2ii2rotinvp(T_unif(5:9, :)); T_unif = cat(1, T_unif, T_unif_p2);
P_unif_p2   = nsmr_T_p2ii2rotinvp(P_unif(5:9, :)); P_unif = cat(1, P_unif, P_unif_p2);
%
T_brain_p2 = nsmr_T_p2ii2rotinvp(T_brain(5:9, :)); T_brain = cat(1, T_brain, T_brain_p2);
P_brain_p2 = nsmr_T_p2ii2rotinvp(P_brain(5:9, :)); P_brain = cat(1, P_brain, P_brain_p2);

T_v    = [ 1       2          3         4       10       11        12 ];
label  = {'N',    'S',       'Z',      'Z',     'S',     'Z',      'T'};
T_lims = [ 0    0.0*1e-9   0.0*1e-9   -0.5      0.0      0.0       0.0 ; ...
           1    1.5*1e-9   4.0*1e-9    1.0      .30       1        1.0];

Ticks     = {0:.25:1; (0:.5:1.5)*1e-9; (0:1:4)*1e-9; -.5:.5:1; 0:.1:.3; 0:.25:1; 0:.25:1};
Ticks_lbl = {{'0', '', '0.5', '', '1'}; 0:.5:1.5; 0:1:4; -.5:.5:1; 0:.1:.3; ...
    {'0', '', '0.5', '', '1'}; {'0', '', '0.5', '', '1'}};

col_unif      = [0.2847    0.6427    0.8326];
col_unif_bad  = [0.9059    0.5098    0.4040];
col_brain     = [0.0235    0.1804    0.3804];
col_brain_bad = [0.9059    0.5098    0.4040];
%
sc = 1;
w_thresh  = .15;
face_alph = .5;
edge_alph = .6;
point_sz  = sc*.25;
lw_axes   = sc*1;
lw_plot   = sc*1;
fs_axes   = sc*8;
tick_l    = .03*[1 1];
%
as.nr     = 3;
as.nc     = 4;
as.l_marg = .2;
as.r_marg = 0;
as.u_marg = 0;
as.b_marg = .15;
as.blk_sp = .0125;
as.w_sc   = .75;
as.h_sc   = .75;
style     = 'cols';
papersize = [20 20];
ext       = '-dpng';
res       = '-r600';

% Show correlation between prediction and target/truth
figure(2), clf

for i = 1:numel(T_v)
    
    feat    = T_v(i);
    
    axh = jm_sub_axh(as, i, style);
%     axh = axes('position',[left bottom width height]);
    hold(axh, 'on')    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Rand features
    
    x         = T_unif(feat, :);
    y         = P_unif(feat, :);
    %
    col = bsxfun(@times, ones(size(T_unif, 2), 1), col_unif);
    if label{i} == 'S'
        ind         = T_unif(1, :) < w_thresh;
        col(ind, :) = repmat(col_unif_bad, sum(ind), 1); 
    elseif label{i} == 'Z'
        ind         = 1 - T_unif(1, :) < w_thresh;
        col(ind, :) = repmat(col_unif_bad, sum(ind), 1); 
    elseif label{i} == 'T'
        ind         = T_unif(1, :) < w_thresh & T_unif(4, :).^2 < .15;
        col(ind, :) = repmat(col_unif_bad, sum(ind), 1);
    else
        ind         = false(size(x));
    end
    %
    [axh, ph] = mplot_corr(x, y, col, axh);
    set(ph, 'MarkerFaceAlpha', face_alph, 'MarkerEdgeAlpha', edge_alph, 'SizeData', point_sz)
    ms               = nsmr_get_perf_metrics( x(~ind), y(~ind));
    ms_rand.r (i)    = ms.r;
    ms_rand.NRMSE(i) = ms.NRMSE;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % In vivo features
    
    x         = T_brain(feat, mask_bol);
    y         = P_brain(feat, mask_bol);
    %
    col = bsxfun(@times, ones(size(T_brain(:, mask_bol), 2), 1), col_brain);
    if label{i} == 'S'
        ind         = T_brain(1, mask_bol) < w_thresh;
        col(ind, :) = repmat(col_brain_bad, sum(ind), 1); 
    elseif label{i} == 'Z'
        ind         = 1 - T_brain(1, mask_bol) < w_thresh;
        col(ind, :) = repmat(col_brain_bad, sum(ind), 1);
    elseif label{i} == 'T'
        ind         = T_brain(1, mask_bol) < w_thresh & T_brain(4, mask_bol).^2 < .16;
        col(ind, :) = repmat(col_brain_bad, sum(ind), 1);
    else
        ind         = false(size(x));
    end        
    [axh, ph] = mplot_corr(x, y, col, axh);
    set(ph, 'MarkerFaceAlpha', face_alph, 'MarkerEdgeAlpha', edge_alph, 'SizeData', point_sz)
    ms             = nsmr_get_perf_metrics( x(~ind), y(~ind));
    ms_WM.r(i)     = ms.r;
    ms_WM.NRMSE(i) = ms.NRMSE;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1-to-1 correlation line
    
    x_regr = [-1 1]*5;
    plot(x_regr, x_regr, 'w-', 'LineWidth', lw_plot)
    plot(x_regr, x_regr, 'k--', 'LineWidth', lw_plot)
    
    hold(axh, 'off')
    set(axh, 'YLim', [T_lims(1, i) T_lims(2, i)], 'XLim', [T_lims(1, i) T_lims(2, i)])
    set(axh, 'XTick', Ticks{i}, 'XTickLabel', Ticks_lbl{i})
    set(axh, 'YTick', Ticks{i}, 'YTickLabel', Ticks_lbl{i})
    set(axh, 'TickDir', 'out', 'TickLength', tick_l, 'Box', 'off')
    set(axh, 'LineWidth', lw_axes, 'FontSize', fs_axes)
            
end

fig_fn = fullfile(dir_path, 'Figures', 'Corr_Plots'); mkdir(fig_fn);
fig_fn = fullfile(fig_fn, 'smr_corr');
fig    = jm_save_fig(papersize, fig_fn, ext, res);