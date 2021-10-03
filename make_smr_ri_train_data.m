clc
close all;
clear;

dataset_pars.T_name = { 's0'    'fs'    'di_s'   'di_z'   'dd_z'   'p20'  'p21r'  'p21i'  'p22r'  'p22i'    't2_s'  't2_z'};
S_norm_method       = 'median';
snr_max = 160; snr_min = 80;
dataset_pars.snr = [snr_max snr_min];
namy = 'train_ri_optimized_182183'; % name given to train dataset

% Training data specs
n_tot       = 590*1e3; % % n_tot (number of training vectors)
ntot_str    = ['_' num2str(n_tot / 1e3)];
f_brain_v   = .5; % fraction of m_brain parameter vectors
n_fit_max   = 12000*1e3; % maximum number of m_fit vectors

% Paths to in vivo dataset fitted with the SMR model
dir_path    = 'D:\Users\joao\Data\Bjorns_Paper\';
invivo_fn_v = {'113_FysikDiffusion_182', '113_FysikDiffusion_183'}; % invivo datasets (MUST HAVE SAME XPS!!!)
invivo_fn_v = fullfile(dir_path, invivo_fn_v);

% Output path for training dataset
o_path = pwd;
o_path = fullfile(o_path, 'Train_data');


dataset_pars.mut_std    = .3; % mutation 'strength' (std of X in Eq. 15)
dataset_pars.mfs_fn     = fullfile(invivo_fn_v, 'NII_RES', 'dtd_smr', 'mfs.mat');
dataset_pars.add_orig   = false; % Add experimentally measured signal data (not from forward model)
dataset_pars.add_mutate = false; % Augment m_brain dataset w/ mutations
check_targets           = true;
check_mask              = false;
add_noise               = true;

n_repeat = 1;

%% Load fit data and xps structure

i_data = fullfile(invivo_fn_v{1}, 'output');
%
xps                = mdm_xps_load(fullfile(i_data, 'dmri_mc_topup_gibbs_xps.mat'));
[ ~, ~, xps.s_ind] = uniquetol([xps.b xps.b_delta xps.te], .01, 'ByRows', true);
xps_pa             = mdm_xps_pa(xps);
indx_norm          = xps_pa.te*1e3 < 65 & xps_pa.b*1e-9 < .11; % indx for signal normalization
%
% xps       = mdm_xps_load(fullfile(fileparts(dir_path), 'xps_alt', 'LTE_only_xps.mat'));
% indx_norm = xps.te*1e3 < 60 & xps.b*1e-9 < .1;

dataset_pars.mask = [];
S_data = []; T_data = [];
%
for invivo_fn = invivo_fn_v
    
    dir_path = invivo_fn{1};
    i_data   = fullfile(dir_path, 'output');
    nii_fn   = fullfile(i_data, 'dmri_mc_topup_gibbs.nii.gz');
    
    % Load mfs structure
    mfs    = mdm_mfs_load(fullfile(dir_path, 'NII_RES', 'dtd_smr', 'mfs.mat'));
    m_temp = mfs.m; m_temp = m_temp(:,:,:,1:(end-1)); % remove entries corresponding to fit residuals
    
    % Load or create skull mask
    mask_4d  = nsmr_make_mask(nii_fn, fullfile(dir_path, 'NII_RES', 'dtd_smr', 'mfs.mat'), 0);
    mask_bol = logical(mask_4d(:));
    
    % plot masked regions
    if check_mask == 1
        figure(1), clf
        im2d = mask_4d .* m_temp(:,:,:,1);
        maps_grid_plot(im2d);
        
        pause(.5)
    end
    
    % Generate synthetic dataset from experimental m structure
    [S_temp, T_temp] = nsmr_train_data_from_m(m_temp, xps);
    
    % Remove voxels in extra-meningeal areas
    S_temp(:,~mask_bol) = []; T_temp(:,~mask_bol) = [];
    
    if dataset_pars.add_orig
        orig_flag = '_orig';
        
        % Load signal data
        [I,~]  = mdm_nii_read(nii_fn); % Load data
        S_orig = nsmr_reshape_4to2D(I);
        S_temp = double(S_orig'); S_temp(:, ~mask_bol) = [];
    end
    
    % Shuffle dataset
    indx = randperm(size(S_temp, 2));
    %
    S_temp = S_temp(:, indx);
    T_temp = T_temp(:, indx);
    
    S_data = cat(2, S_data, S_temp);
    T_data = cat(2, T_data, T_temp);
    
    clear m_temp S_temp T_temp
end


%%
for f_brain = f_brain_v
    
    train_rep = cell(1, n_repeat);
    dataset_pars.add_invivo = logical(f_brain);
    
    for i = 1:n_repeat
        
        S_brain = []; S_mut = []; S_unif = [];
        T_brain = []; T_mut = []; T_unif = [];
        orig_flag = []; noise_flag = [];
        
        if dataset_pars.add_invivo || dataset_pars.add_mutate            
            
            % Select m_fit parameter vectors
            if n_fit_max < size(S_data, 2)
                S_brain = S_data(:, 1:n_fit_max);
                T_brain = T_data(:, 1:n_fit_max);
            else
                S_brain = S_data;
                T_brain = T_data;
            end
            
            % Define number of invivo training datapoints
            n_brain = round(f_brain * n_tot);
            if n_brain > size(S_brain, 2)
                dataset_pars.n_mut = n_brain - size(S_brain, 2);
                dataset_pars.add_mutate = true;
                
                % Complement m_brain dataset with m_mut parameter vectors
                [T_mut, S_mut] = nsmr_mutate_train_data( T_brain, xps, dataset_pars.n_mut, dataset_pars.mut_std);
                
                S_brain = cat(2, S_brain, S_mut);
                T_brain = cat(2, T_brain, T_mut);
            else
                indx = randperm(size(S_brain, 2));
                S_brain = S_brain(:, indx(1:n_brain));
                T_brain = T_brain(:, indx(1:n_brain));
            end
        end
        
        dataset_pars.n_brain = size(S_brain, 2);
        dataset_pars.n_unif = n_tot - size(S_brain, 2);
        dataset_pars.add_rand = logical(dataset_pars.n_unif);
        
        if dataset_pars.add_rand
            % Generate random synthetic dataset
            maxp = sqrt(5 / (4*pi));
            T_lims = [ 0.5   0    0.07*1e-9   0.2*1e-9   -0.46   -maxp    -maxp    -maxp    -maxp     -maxp     .03     .03; ...
                4    1    1.33*1e-9    4*1e-9     0.86    maxp     maxp     maxp     maxp      maxp     .30      1 ];
            [S_unif, T_unif, ~, ~, ~] = nsmr_train_data_from_rand(xps, dataset_pars.n_unif, T_lims);
        end
        
        %% Merge the various training datasets
        
        S_train = cat(2, S_brain, S_unif);
        T_train = cat(2, T_brain, T_unif);
        
        % Shuffle data
        indx = randperm(size(S_train, 2));
        S_train = S_train(:, indx);
        T_train = T_train(:, indx);
                
        % Add Rician distributed noise
        if add_noise           
            
            % Add noise
            noise_flag = '_noisy';
            snr        = unifrnd(snr_min, snr_max, 1, size(T_train,2));
            S_train    = dd_get_rice_noise(S_train, snr, T_train(1, :));
            
        end
        
        % Project into SH basis
        S0_train = zeros(max(xps.s_ind), size(S_train, 2)); S2_train = S0_train;
        tic
        parfor n_vxl = 1:size(S_train, 2)
            [S0_train(:, n_vxl), S2_train(:, n_vxl), ~] =...
                smr_project_ri(S_train(:,n_vxl), xps, 0);
        end
        toc
        
        m_train = nsmr_reshape_2to4D(T_train, [size(T_train, 2) 1 1]);
        m_train = nsmr_m2rotinvm(m_train);
        m_train = msf_notfinite2zero(m_train);
        %
        T_train = nsmr_reshape_4to2D(m_train);
        
        % Normalize Signal
        [S0_train, S0_norm_cnst] = nsmr_normalize_signal(S0_train, indx_norm, S_norm_method);
        S2_train = bsxfun( @rdivide, S2_train, S0_norm_cnst);
        
        % Store signal normalization details in a structure
        % S0
        dataset_pars.s0_norm_pars.function = 'nsmr_normalize_signal';
        dataset_pars.s0_norm_pars.function_input{1}  = indx_norm;
        dataset_pars.s0_norm_pars.function_input{2}  = S_norm_method;
        dataset_pars.s0_norm_pars.function_output{1} = S0_norm_cnst;
        % S2
        dataset_pars.s2_norm_pars.function = 'nsmr_normalize_signal';
        dataset_pars.s2_norm_pars.function_input{1}  = indx_norm;
        dataset_pars.s2_norm_pars.function_input{2}  = S_norm_method;
        dataset_pars.s2_norm_pars.function_output{1} = S0_norm_cnst;
        
        % Remove constant rows
        [S0_train, ~] = removeconstantrows(S0_train);
        
        % Define fractions of mutated and random datasets relative to the total
        % number of training data points
        % Example -> frac = [frac_invivo frac_rand] = [.2 .8]
        dataset_pars.frac = [size(T_brain, 2)/size(T_train, 2) size(T_unif, 2)/size(T_train, 2)];
        dataset_pars.frac = round(dataset_pars.frac, 2);
        
        % Store training parameters in cell entry
        train_rep{i}.xps          = xps;
        train_rep{i}.xps_pa       = xps_pa;
        train_rep{i}.S0_train     = S0_train;
        train_rep{i}.S2_train     = S2_train;
        train_rep{i}.T_train      = T_train;
        train_rep{i}.dataset_pars = dataset_pars;
        
        
        %% Check Target params
        if check_targets
            figure(1), clf
            dd_plot_target( T_train, dataset_pars.T_name);
            sgtitle('T_{train}')
        end
        
    end
    
    %% Save Training dataset
    train_fn = [namy '_ri' ntot_str orig_flag '_' num2str(dataset_pars.frac(1), 2) '_' ...
        num2str(dataset_pars.frac(2), 2) noise_flag '.mat'];
    
    o_fn   = fullfile(o_path, train_fn);
    save(o_fn, 'train_rep', '-v7.3')
    
end

