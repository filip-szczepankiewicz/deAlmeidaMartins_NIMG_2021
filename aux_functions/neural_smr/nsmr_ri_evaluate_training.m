clear; close all;

net_dir = 'D:\Users\joao\Data\Neural_RI_SMR\';
invivo_dir = '113_FysikDiffusion_185'; % Select in vivo dataset for comparison


n_nets = numel(net_name);

do_mask = 1; % Focus on WM
only_s0 = 0;

%% Load in vivo data and SM% Define network names
f_v = [0:.05:.45 .48]; n_tot_str = '_200';
net_name = cell(1, length(f_v));
for i = 1:numel(f_v)
    f = f_v(i);
    net_name{1, i} = ['net_RI' n_tot_str '_' num2str(f) '_0_' num2str(1-f) '_noisy.mat'];
endR fit results

% Connect to data
data_dir = fullfile('D:\Users\joao\Data\Bjorns_Paper', invivo_dir);
i_data = fullfile(data_dir, 'output'); 
nii_fn = fullfile(i_data, 'dmri_mc_topup_gibbs.nii.gz');
[I,~]  = mdm_nii_read(nii_fn); % Load data

% Create data linked to the features we need to recover
S_invivo = nsmr_reshape_4to2D(I);
clear I

% Load xps
net_fn = fullfile(net_dir, 'Train_Networks', net_name{1});
load(net_fn); net = nets{end}; xps = net.userdata.xps;

% Project into SH basis
S0_invivo = zeros(max(xps.s_ind), size(S_invivo, 2)); S2_invivo = S0_invivo; 
tic
parfor i = 1:size(S_invivo,2)
    [S0_invivo(:, i), S2_invivo(:, i), ~] = smr_project_ri(S_invivo(:,i), xps, 0);
end
toc

% Normalize signal
norm_function = net.userdata.dataset_pars.s0_norm_pars.function;
inputs_s0 = net.userdata.dataset_pars.s0_norm_pars.function_input; 
eval(['[S0_invivo, S0_norm_cnst] = ' norm_function '( S0_invivo, inputs_s0{1}, inputs_s0{2});'])
S2_invivo = bsxfun( @rdivide, S2_invivo, S0_norm_cnst);
% Remove constant rows
[S0_invivo, ~] = removeconstantrows(S0_invivo);
if only_s0
    S_invivo = S0_invivo;
else
    S_invivo = cat(1, S0_invivo, S2_invivo);
end

% Load Fitting results for comparison
i_fit = fullfile(data_dir, 'NII_RES','dtd_smr'); 
mfs_fn = fullfile(i_fit,'mfs.mat'); 
mfs = mdm_mfs_load(mfs_fn); % Load mfs structure
m = mfs.m(:, :, :, 1:(end-1));

% calculate Rot Inv model parameters
m_ri = nsmr_m2rotinvm(m);

% T_invivo is the target (label)
T_invivo = nsmr_reshape_4to2D(m_ri);

%% Mask out non-brain areas (skull_mask) and non-WM areas (mask)
if do_mask
    % Create uFA mask
    paths.uFA_nii = fullfile(data_dir, 'NII_RES', 'maps', 'dtd_covariance_uFA.nii.gz');
    [I_uFA, ~]  = mdm_nii_read(paths.uFA_nii); % Load uFA maps
    uFA_mask = I_uFA > .6;
    
    % Create skull mask
    paths.s0_nii = fullfile(data_dir, 'NII_RES', 'maps', 'dtd_covariance_s0.nii.gz');
    [I_s0, ~]  = mdm_nii_read(paths.s0_nii); % Load s_0 maps
    skull_mask = jm_skull_strip(I_s0, .1*max(I_s0(:)), 4, 2, 0);
    skull_mask_bol = logical(skull_mask(:));
    
    mask = uFA_mask .* skull_mask;
    
    % Remove top and bottom slices
    mask(:, :, 1) = 0*mask(:, :, 1); mask(:, :, end) = 0*mask(:, :, end);
    mask_bol = logical( mask(:) );
else
    mask = ones( size(m_ri, 1:3) );
    mask_bol = logical( mask(:) );
end

%% Evaluate performance of the various networks

train.perf = []; train.r = []; train.m = []; train.b = [];
val = train; invivo = train;

for i = 1:numel(net_name)
    
    % Load previously trained network
    net_fn = fullfile(net_dir, 'Train_Networks', net_name{i});
    load(net_fn)
    net = nets{end};
    T_ind = net.userdata.T_ind; n_ind = numel(T_ind);
    T_name = net.userdata.dataset_pars.T_name;       
    
    % Training metrics
    train.perf = cat(2, train.perf, net.userdata.test_train.perf);
    train.r    = cat(2, train.r, net.userdata.test_train.r);
    train.m    = cat(2, train.m, net.userdata.test_train.m);
    train.b    = cat(2, train.b, net.userdata.test_train.b);
    
    % Validation metrics
    val.perf   = cat(2, val.perf, net.userdata.test_val.perf);
    val.r      = cat(2, val.r, net.userdata.test_val.r);
    val.m      = cat(2, val.m, net.userdata.test_val.m);
    val.b      = cat(2, val.b, net.userdata.test_val.b);
    
    % In vivo metrics
    P_invivo = net(S_invivo); % Predict data
    T_ind = net.userdata.T_ind;
    [T_invivo_norm, ~] = nsmr_feature_norm( T_invivo(T_ind, :), ... % Feature normalization
        net.userdata.feat_norm_pars.method, net.userdata.feat_norm_pars); 
    % Performance metrics
    invivo_temp = nsmr_get_perf_metrics( T_invivo_norm(:, mask_bol), P_invivo(:, mask_bol));
    
    invivo.perf = cat(2, invivo.perf, invivo_temp.perf);
    invivo.r    = cat(2, invivo.r, invivo_temp.r);
    invivo.m    = cat(2, invivo.m, invivo_temp.m);
    invivo.b    = cat(2, invivo.b, invivo_temp.b);    
    
end



%% Plot risk curves

% figure(1), clf
% 
% axh = gca;
% hold(axh, 'on')
% p1 = plot( f_v, train.perf, 'b-');
% p2 = plot( f_v, val.perf, 'r-');
% p3 = plot( f_v, invivo.perf, '-', 'Color', [0 .7 0]);
% hold(axh, 'off')
% ylim([0 .1])
% 
% legend([p1 p2 p3], {'Train', 'Valid', 'In Vivo'})

%% Plot regression metrics

% Train
figure(2), clf
axh = dd_plot_features(invivo.r, 1, T_name(T_ind), f_v);

figure(3), clf
axh = dd_plot_features(invivo.m, 1, T_name(T_ind), f_v);

figure(4), clf
axh = dd_plot_features(invivo.perf, 0, T_name(T_ind), f_v);

