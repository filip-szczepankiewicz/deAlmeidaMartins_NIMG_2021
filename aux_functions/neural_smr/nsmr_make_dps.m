clear; close all;

do_mask    = 1; % Mask skull
do_WM_mask = 0; % Focus on WM

net_name   = 'deep_net_LOO_s0snr_5900_0.48_0.52_noisy'; % Select network net_multi_f_590_0.34_0.66_noisy_rep_5
invivo_dir = '113_FysikDiffusion_185'; % In Vivo dataset to deploy the NN
%
net_dir    = 'D:\Users\joao\Data\Neural_SMR\';
net_fn     = fullfile(net_dir, 'Train_Networks', [net_name '.mat']);

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



%% Performance on in vivo data

% Connect to data
data_dir = fullfile('D:\Users\joao\Data\Bjorns_Paper', invivo_dir);
nii_fn   = fullfile(data_dir, 'output', 'dmri_mc_topup_gibbs.nii.gz');

% Load LSQ fit results
mfs_fn = fullfile(data_dir, 'NII_RES', 'dtd_smr', 'mfs.mat'); 
mfs    = mdm_mfs_load(mfs_fn); % Load mfs structure
m      = mfs.m(:,:,:,1:(end-1));
sz     = size(m);

[I, ~] = mdm_nii_read(nii_fn); % Load data
S_exp  = nsmr_reshape_4to2D( double(I) ); clear I

% Normalize signal
eval(['[S_exp, ~] = ' norm_function '( S_exp, inputs{1}, inputs{2});'])

tic
% Predict data
if contains(net_name, 'deep')
    P_exp = predict(net, S_exp');
    P_exp = P_exp';
else
    P_exp = net(S_exp);
end

% Convert predictions to SI units
[P_exp, ~] = nsmr_feature_undo_norm( P_exp, feat_norm_pars);
toc


m_exp  = nsmr_reshape_2to4D(P_exp, sz(1:3));
m_exp  = cat(4, ones(sz(1:3)), m_exp);

% SMR parameters
s0   = m_exp(:,:,:,1);
fs   = m_exp(:,:,:,2);
di_s = m_exp(:,:,:,3);
di_z = m_exp(:,:,:,4);
dd_z = m_exp(:,:,:,5);
p20  = m_exp(:,:,:,6);
p21r = m_exp(:,:,:,7);
p21i = m_exp(:,:,:,8);
p22r = m_exp(:,:,:,9);
p22i = m_exp(:,:,:,10);
t2_s = m_exp(:,:,:,11);
t2_z = m_exp(:,:,:,12);

out_fn = fullfile(data_dir, 'NII_RES', 'dtd_smr', 'dps_deep_net.mat');
save(out_fn, 'm_exp', 's0', 'fs', 'di_s', 'di_z', 'dd_z', ...
    'p20', 'p21r', 'p21i', 'p22r', 'p22i', 't2_s', 't2_z');

