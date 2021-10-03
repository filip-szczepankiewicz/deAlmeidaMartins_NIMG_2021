clc
close all;
clear;

dataset_pars.T_name = { 's0'    'fs'    'di_s'   'di_z'   'dd_z'  't2_s'  't2_z'  'p2'};
snr_max = 50; snr_min = 20;
dataset_pars.snr = [snr_max snr_min];
S_norm_method = 'median';

dataset_pars.n_rand = 750*1e3;
dataset_pars.n_mut = 0*1e3;

dataset_pars.add_invivo = 0;
dataset_pars.add_orig = 0;
dataset_pars.add_mutate = logical(dataset_pars.n_mut);
dataset_pars.add_rand = logical(dataset_pars.n_rand);

check_mask = false;
add_noise = true;

%% Generate random synthetic dataset

S_basis = []; S_mut = []; S_rand = [];
T_basis = []; T_mut = []; T_rand = [];
orig_flag = []; noise_flag = [];

%% Prepare paths and load data

dir_path = 'D:\Users\joao\Data\Bjorns_Paper\113_FysikDiffusion_182';
i_data = fullfile(dir_path, 'output');
i_fit = fullfile(dir_path, 'NII_RES');

% Connect to fit results
model = 'dtd_smr'; % Choose model
dataset_pars.mfs_fn = fullfile(i_fit, model, 'mfs.mat');

% Load xps structure
xps = mdm_xps_load(fullfile(i_data, 'dmri_mc_topup_gibbs_xps.mat'));
[ ~, ~, xps.s_ind] = uniquetol([xps.b xps.b_delta xps.te], .01, 'ByRows', true);
xps_pa = mdm_xps_pa(xps);

% Load mfs structure
mfs = mdm_mfs_load(dataset_pars.mfs_fn);
m = mfs.m;
m = m(:,:,:,1:(end-1)); % remove entries corresponding to fit residuals

%% Generate training dataset from in vivo results

S_basis = []; S_mut = []; S_rand = [];
T_basis = []; T_mut = []; T_rand = [];
orig_flag = []; noise_flag = [];

if dataset_pars.add_invivo || dataset_pars.add_mutate
    
    % Create mask to exclude non-brain areas
    s0 = mfs.m(:,:,:,1);
    s0_mask = s0 > .05*max(s0(:));
    skull_mask = jm_skull_strip(s0, .075*max(s0(:)), 3, 2, 0);
    
    dataset_pars.mask = s0_mask.*skull_mask;
    % Mask-out top and bottom slices
    dataset_pars.mask(:,:,1) = 0*dataset_pars.mask(:,:,1); dataset_pars.mask(:,:,end) = 0*dataset_pars.mask(:,:,end);
    
    mask_bol = logical(dataset_pars.mask(:));
    
    if check_mask == 1
        
        figure(1), clf
        im2d = dataset_pars.mask.*m(:,:,:,1);
        maps_grid_plot(im2d);
        
    end  

    % calculate Rot Inv model parameters
    m_ri = nsmr_m2rotinvm(m);
    
    % Generate synthetic dataset from experimental m structure
    [S_basis, T_basis] = nsmr_ri_train_data_from_m(m_ri, xps_pa);
    % Remove voxels in extra-meningeal areas
    S_basis(:,~mask_bol) = []; T_basis(:,~mask_bol) = [];
    
    if dataset_pars.add_orig
        
        orig_flag = '_orig';
        
        % Load signal data
        nii_fn = fullfile(i_data, 'dmri_mc_topup_gibbs.nii.gz');
        [I,~]  = mdm_nii_read(nii_fn); % Load data
        sz = size(I);
        S_orig = reshape(I, prod(sz(1:3)), sz(4));
        S_orig = double(S_orig'); S_orig(:,~mask_bol) = [];
        
        % Project into SH basis
        S0_orig = zeros(max(xps.s_ind), size(S_orig, 2)); S2_orig = S0_orig;
        tic
        parfor i = 1:size(S_orig,2)
            [S0_orig(:, i), S2_orig(:, i), ~] = smr_project_ri(S_orig(:,i), xps, 0);
        end
        toc
        S_basis = cat(1, S0_orig, S2_orig);
                       
    end
    
    if dataset_pars.add_mutate
        
        % Generate synthetic dataset by mutating a subset of m-derived datapoints
        [T_mut, S_mut] = nsmr_ri_mutate_train_data( T_basis, xps_pa, dataset_pars.n_mut, .2);        
        
    end
end

if dataset_pars.add_rand
%               1      2         3            4           5      6       7      8 
%               s0    fs      di_s           di_z       dd_z    t2_s    t2_z    p2  
    T_lims = [ 0.5     0    0.07*1e-9     0.2*1e-9     -0.46    .03     .03     0; ...
                4      1    1.33*1e-9      4*1e-9       0.86    .30      1      1];
    [S_rand, T_rand, ~, ~, ~] = nsmr_ri_train_data_from_rand(xps_pa, dataset_pars.n_rand, T_lims);
    
end

%% Merge the various training datasets

S_train = cat(2, S_basis, S_mut, S_rand);
T_train = cat(2, T_basis, T_mut, T_rand);

% Shuffle data
indx = randperm(size(S_train, 2));
S_train = S_train(:, indx);
T_train = T_train(:, indx);

% Separate S into S0 and S2
S0_train = S_train(1:(end/2), :);
S2_train = S_train((end/2 + 1):end, :);

% Add Rician distributed noise
if add_noise
    
    % Normalize Signal
    indx = xps_pa.te*1e3 < 65 & xps_pa.b*1e-9 < .11;
    [S0_train, S0_norm_cnst] = nsmr_normalize_signal(S0_train, indx, S_norm_method);
    S2_train = bsxfun( @rdivide, S2_train, S0_norm_cnst);
    
    noise_flag = '_noisy';
    snr = unifrnd(snr_min, snr_max, 1, size(T_train, 2));
    S_train = dd_get_rice_noise( cat(1, S0_train, S2_train), snr);
    
    % Undo Signal normalization
    S_train = bsxfun( @times, S_train, S0_norm_cnst);
    
    % Separate S into S0 and S2
    S0_train = S_train(1:size(S0_train,1), :);
    S2_train = S_train((size(S0_train,1) + 1):end, :);
    
end

% Normalize Signal
indx = xps_pa.te*1e3 < 65 & xps_pa.b*1e-9 < .11;
[S0_train, S0_norm_cnst] = nsmr_normalize_signal(S0_train, indx, S_norm_method);
S2_train = bsxfun( @rdivide, S2_train, S0_norm_cnst);

% Store signal normalization details in a structure
% S0
dataset_pars.s0_norm_pars.function = 'nsmr_normalize_signal';
dataset_pars.s0_norm_pars.function_input{1} = indx;
dataset_pars.s0_norm_pars.function_input{2} = S_norm_method;
dataset_pars.s0_norm_pars.function_output{1} = S0_norm_cnst;
% S2
dataset_pars.s2_norm_pars.function = 'nsmr_normalize_signal';
dataset_pars.s2_norm_pars.function_input{1} = indx;
dataset_pars.s2_norm_pars.function_input{2} = S_norm_method;
dataset_pars.s2_norm_pars.function_output{1} = S0_norm_cnst;

% Remove constant rows
[S0_train, PS] = removeconstantrows(S0_train);

%% Check Target params

figure(1), clf
dd_plot_target( T_train, dataset_pars.T_name);
sgtitle('T_{train}')


%% Save Training dataset

% Define fractions of mutated and random datasets relative to the total 
% number of training data points
% Example -> frac = [frac_invivo frac_mut frac_rand] = [.2 .3 .5]
dataset_pars.frac = [size(T_basis, 2)/size(T_train, 2) size(T_mut, 2)/size(T_train, 2) ...
    size(T_rand, 2)/size(T_train, 2)];
dataset_pars.frac = round(dataset_pars.frac, 2);

train_fn = ['train_RI' orig_flag '_' num2str(dataset_pars.frac(1), 2) '_' num2str(dataset_pars.frac(2), 2)...
    '_' num2str(dataset_pars.frac(3), 2) noise_flag '.mat'];

o_path = 'D:\Users\joao\Data\Neural_RI_SMR';
o_fn = fullfile(o_path, 'Train_data', train_fn);
save(o_fn, 'xps', 'S0_train', 'S2_train', 'T_train', 'dataset_pars')

