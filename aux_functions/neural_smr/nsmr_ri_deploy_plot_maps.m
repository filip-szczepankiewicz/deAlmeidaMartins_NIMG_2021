clear; close all;

do_mask    = 1; % Mask skull
do_WM_mask = 0; % Focus on WM

net_name   = 'net_LOO_ri_590_0.48_0.52_noisy'; % Select network
invivo_dir = '113_FysikDiffusion_185'; % In Vivo dataset to deploy the NN
%
net_dir    = 'D:\Users\joao\Data\Neural_RI_SMR';
net_fn     = fullfile(net_dir, 'Train_Networks', [net_name '.mat']);
only_s0    = logical( strfind(net_name, '_s0'));

% Load previously trained network
load(net_fn)
%
if contains(net_name, 'deep')
    xps    = userdata.xps;
    xps_pa = userdata.xps_pa;
    T_ind  = userdata.T_ind;
    T_name = userdata.dataset_pars.T_name;
    %
    feat_norm_pars   = userdata.feat_norm_pars;
    norm_function_s0 = userdata.dataset_pars.s0_norm_pars.function;
    inputs_s0        = userdata.dataset_pars.s0_norm_pars.function_input;
    norm_function_s2 = userdata.dataset_pars.s2_norm_pars.function;
    inputs_s2        = userdata.dataset_pars.s2_norm_pars.function_input;
    %
    snr_max = max(userdata.dataset_pars.snr);
    snr_min = min(userdata.dataset_pars.snr);
else
    net    = nets{end}; clear nets
    xps    = net.userdata.xps;
    xps_pa = net.userdata.xps_pa;
    T_ind  = net.userdata.T_ind;
    T_name = net.userdata.dataset_pars.T_name;
    %
    feat_norm_pars   = net.userdata.feat_norm_pars;
    norm_function_s0 = net.userdata.dataset_pars.s0_norm_pars.function;
    inputs_s0        = net.userdata.dataset_pars.s0_norm_pars.function_input;
    norm_function_s2 = net.userdata.dataset_pars.s2_norm_pars.function;
    inputs_s2        = net.userdata.dataset_pars.s2_norm_pars.function_input;
    %
    snr_max = max(net.userdata.dataset_pars.snr);
    snr_min = min(net.userdata.dataset_pars.snr);
end

%% Performance on in vivo data

% Connect to data
data_dir = fullfile('D:\Users\joao\Data\Bjorns_Paper', invivo_dir);
nii_fn   = fullfile(data_dir, 'output', 'dmri_mc_topup_gibbs.nii.gz');

% Load LSQ fit results
mfs_fn = fullfile(data_dir, 'NII_RES', 'dtd_smr', 'mfs.mat');
mfs    = mdm_mfs_load(mfs_fn); % load mfs structure
m      = mfs.m(:,:,:,1:(end-1)); % remove fit residuals
m_ri   = nsmr_m2rotinvm(m); % convert to RotInv parameters


%% Create data linked to the features we need to recover

% Generate synthetic dataset from experimental m_ri structure
[S_brain, T_brain] = nsmr_ri_train_data_from_m(m_ri, xps_pa);

% Separate S_invivo into S0 and S2
S0_brain = S_brain(1:(end/2), :);
S2_brain = S_brain((end/2 + 1):end, :);

% Add noise
snr      = unifrnd(snr_min, snr_max, 1, size(S0_brain, 2));
S0_brain = dd_get_rice_noise( S0_brain, snr, T_brain(1, :));
%
snr      = unifrnd(snr_min, snr_max, 1, size(S2_brain, 2));
S2_brain = dd_get_rice_noise( S2_brain, snr, T_brain(1, :));

% Normalize signal
eval(['[S0_brain, S0_norm_cnst] = ' norm_function_s0 '( S0_brain, inputs_s0{1}, inputs_s0{2});'])
S2_brain = bsxfun( @rdivide, S2_brain, S0_norm_cnst);
%
[S0_brain, ~] = removeconstantrows(S0_brain); % Remove constant rows

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
sz     = size(m_ri); 
%
m_pred = nsmr_reshape_2to4D(P_brain, sz(1:3));
m_pred = cat(4, ones(sz(1:3)), m_pred);
m_pred = msf_notfinite2zero(m_pred);
%
m_exp  = nsmr_reshape_2to4D(P_exp, sz(1:3));
m_exp  = cat(4, ones(sz(1:3)), m_exp);
m_exp  = msf_notfinite2zero(m_exp);

slice = 19;
d     = 3;

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
style     = 'rows';
papersize = [10 10];
ext       = '-dmeta';
res       = 'Inf';

% Maps Standard
for i = 1:numel(T_ind)
    
    figure(2*(i-1) + 1), clf
    
    % Target
    clear im3d
    im3d = (m_targ(:, :, :, T_ind(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 1, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    % Pred
    clear im3d
    im3d = (m_pred(:, :, :, T_ind(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 5, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    % Diff
    clear im3d
    im3d = ( m_pred(:, :, :, T_ind(i)) - m_targ(:, :, :, T_ind(i)) );
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 9, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, mplot_cmaphotcold(30))
    set(axh, 'CLim', [-1 1] * .5 * m_lims(2, i))
    axis(axh,'off')
    
    % Experimental (predicted)
    clear im3d
    im3d = (m_exp(:, :, :, T_ind(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = jm_sub_axh(as, 13, style);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'off')
    
    fig_fn = fullfile(net_dir, 'Figures', 'Maps_standard', m_name{i});
    fig    = jm_save_fig(papersize, fig_fn, ext, res);
    %
    pause(.3)
    
    figure(2*i), clf
    
    % Experimental (predicted)
    clear im3d
    im3d = (m_exp(:, :, :, T_ind(i)) - m_lims(1, i))/(m_lims(2, i) - m_lims(1, i));
    im3d = mask .* im3d;
    %
    axh = axes('Position', [0 0 1 1]);
    msf_imagesc(im3d, d, slice);
    colormap(axh, 'hot')
    set(axh, 'CLim', [0 1])
    axis(axh,'tight','off')
    
    fig_fn = fullfile(net_dir, 'Figures', 'Maps_standard', [m_name{i} '_exp']);
    fig    = jm_save_fig(papersize, fig_fn, ext, res);
    %
    pause(.3)
end