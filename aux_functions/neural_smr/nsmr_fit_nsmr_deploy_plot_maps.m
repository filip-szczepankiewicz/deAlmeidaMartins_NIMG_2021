clear; close all;

do_mask    = 1; % Mask skull
do_WM_mask = 0; % Focus on WM

net_name   = 'net_multi_f_590_0.34_0.66_noisy_rep_5'; % Select network net_multi_f_590_0.34_0.66_noisy_rep_5
invivo_dir = '113_FysikDiffusion_185'; % In Vivo dataset to deploy the NN
%
net_dir    = 'D:\Users\joao\Data\Neural_SMR\';
net_fn     = fullfile(net_dir, 'Train_Networks', [net_name '.mat']);

% Load previously trained network
load(net_fn, 'nets')
net    = nets{end}; clear nets
xps    = net.userdata.xps;
T_ind  = net.userdata.T_ind;
T_name = net.userdata.dataset_pars.T_name;

snr_max = 50; snr_min = 20;

%% Performance on in vivo data

% Connect to data
data_dir = fullfile('D:\Users\joao\Data\Bjorns_Paper', invivo_dir);
nii_fn   = fullfile(data_dir, 'output', 'dmri_mc_topup_gibbs.nii.gz');

% Load LSQ fit results
mfs_fn = fullfile(data_dir, 'NII_RES', 'dtd_smr', 'mfs.mat'); 
mfs    = mdm_mfs_load(mfs_fn); % Load mfs structure
m      = mfs.m(:,:,:,1:(end-1));

norm_function = net.userdata.dataset_pars.s_norm_pars.function;
inputs        = net.userdata.dataset_pars.s_norm_pars.function_input;
%
%% Create data linked to the features we need to recover

% Generate synthetic dataset from experimental m structure
[S_invivo, T_invivo] = nsmr_train_data_from_m(m, xps);

% Normalize signal
eval(['[S_invivo, ~] = ' norm_function '( S_invivo, inputs{1}, inputs{2});'])

% Add noise
snr      = unifrnd(snr_min, snr_max, 1, size(S_invivo, 2));
S_invivo = dd_get_rice_noise(S_invivo, snr);

% Predict data
n_voxels = size(S_invivo, 2);
P_invivo = zeros(13, n_voxels);
opt      = dtd_smr_opt();

parfor i = 1:n_voxels
    if ~any(isnan(S_invivo(:, i)))
        P_invivo(:, i) = dtd_smr_1d_data2fit(S_invivo(:, i), xps, opt);
    end
end

% % Convert predictions to SI units
% [P_invivo, ~] = nsmr_feature_undo_norm( P_invivo(2:(end-1),:), net.userdata.feat_norm_pars);
P_invivo = P_invivo(2:(end-1), :);

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

m_targ = nsmr_m2rotinvm(m);
%
m_pred = nsmr_reshape_2to4D(P_invivo, size(m_targ, 1:3));
m_pred = cat(4, ones(size(m_targ, 1:3)), m_pred);
m_pred = nsmr_m2rotinvm(m_pred);
m_pred = msf_notfinite2zero(m_pred);
%
m_exp  = nsmr_reshape_2to4D(P_invivo, size(m_targ, 1:3));
m_exp  = cat(4, ones(size(m_targ, 1:3)), m_exp);
m_exp  = nsmr_m2rotinvm(m_exp);
m_exp  = msf_notfinite2zero(m_exp);

slice = 19;
d     = 3;

m_v    = [ 2       3          4         5      6      7       8];
label  = {'N',    'S',       'Z',      'Z',   'S',   'Z',    'N'};
m_lims = [ 0    0.0*1e-9   0.0*1e-9   -0.5    0.0    0.0     0.0 ; ...
           1    2.0*1e-9   2.0*1e-9    1.0    .20    .20     1.0];
m_name = {'fs',  'diS',     'diZ',    'ddZ', 't2S',  't2Z',  'p2'};

as.nr     = 4;
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

% Maps Standard
for i = 1:numel(m_v)
    
    figure(1), clf
    
    % Target
    clear im3d
    im3d = (m_targ(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 1, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    % Pred
    clear im3d
    im3d = (m_pred(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 5, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    % Diff
    clear im3d
    im3d = ( m_pred(:, :, :, m_v(i)) - m_targ(:, :, :, m_v(i)) );
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 9, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, mplot_cmaphotcold(30))
    set(axh, 'CLim', [-1 1] * .5 * m_lims(2, i))   
    axis(axh,'off')
    
    % Target
    clear im3d
    im3d = (m_exp(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 13, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    fig_fn = fullfile(net_dir, 'Figures', 'Maps_standard', ['net_' m_name{i}]);
    fig    = jm_save_fig(papersize, fig_fn, ext, res);
    %
    pause(.3)
    
    figure(2), clf
    
    % Target
    clear im3d
    im3d = (m_exp(:, :, :, m_v(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = axes('Position', [0 0 1 1]);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'tight','off')
    
    fig_fn = fullfile(net_dir, 'Figures', 'Maps_standard', ['net_' m_name{i} '_exp']);
    fig    = jm_save_fig(papersize, fig_fn, ext, res);
    %
    pause(.3)
end

