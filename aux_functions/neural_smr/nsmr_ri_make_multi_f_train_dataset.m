clc
close all;
clear;

dataset_pars.T_name = { 's0'    'fs'    'di_s'   'di_z'   'dd_z'   'p20'  'p21r'  'p21i'  'p22r'  'p22i'    't2_s'  't2_z'};
S_norm_method       = 'median';
snr_max = 50; snr_min = 20;
dataset_pars.snr = [snr_max snr_min];

n_tot      = 1180*1e3; ntot_str = ['_' num2str(n_tot / 1e3)];
f_invivo_v = .35 ;
n_repeat   = 1;

dataset_pars.add_orig   = false;
dataset_pars.add_mutate = false;
check_targets           = false;
check_mask              = false;
add_noise               = true;

%% Prepare paths and load data

dir_path = 'D:\Users\joao\Data\Bjorns_Paper\113_FysikDiffusion_182';
i_data   = fullfile(dir_path, 'output');
i_fit    = fullfile(dir_path, 'NII_RES');
nii_fn   = fullfile(i_data, 'dmri_mc_topup_gibbs.nii.gz');

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

% Load or create skull mask
dataset_pars.mask = nsmr_make_mask(nii_fn, dataset_pars.mfs_fn, 0);
mask_bol = logical(dataset_pars.mask(:));

if check_mask == 1
    figure(1), clf
    im2d = dataset_pars.mask.*m(:,:,:,1);
    maps_grid_plot(im2d);
end

for f_invivo = f_invivo_v
    
    train_rep = cell(1, n_repeat);
    dataset_pars.add_invivo = logical(f_invivo);
    
    for i = 1:n_repeat       
      
        S_invivo = []; S_mut = []; S_rand = [];
        T_invivo = []; T_mut = []; T_rand = [];
        orig_flag = []; noise_flag = [];
        
        if dataset_pars.add_invivo || dataset_pars.add_mutate
            
            % calculate Rot Inv model parameters
            m_ri = nsmr_m2rotinvm(m);
                        
            % Generate synthetic dataset from experimental m_ri structure
            [S_invivo, T_invivo] = nsmr_ri_train_data_from_m(m_ri, xps_pa);
            % Remove voxels in extra-meningeal areas
            S_invivo(:,~mask_bol) = []; T_invivo(:,~mask_bol) = [];
            
            if dataset_pars.add_orig           
                orig_flag = '_orig';
                
                % Load signal data
                [I,~]    = mdm_nii_read(nii_fn); % Load data
                S_orig   = nsmr_reshape_4to2D(I);
                S_invivo = double(S_orig); S_invivo(:,~mask_bol) = [];
                
                % Project into SH basis
                S0_invivo = zeros(max(xps.s_ind), size(S_invivo, 2)); S2_invivo = S0_invivo;
                tic
                parfor n_vxl = 1:size(S_invivo, 2)
                    [S0_invivo(:, n_vxl), S2_invivo(:, n_vxl), ~] =...
                        smr_project_ri(S_invivo(:,n_vxl), xps, 0);
                end
                toc
                S_invivo = cat(1, S0_invivo, S2_invivo);
            end
            
            % Define number of invivo training datapoints
            n_invivo = round(f_invivo * n_tot);
            if n_invivo > size(S_invivo, 2)
                dataset_pars.n_mut = n_invivo - size(S_invivo, 2);
                dataset_pars.add_mutate = true;
                
                % Generate synthetic dataset by mutating a subset of m-derived datapoints
                [T_mut, S_mut] = nsmr_ri_mutate_train_data( T_invivo, xps_pa, dataset_pars.n_mut, .5);
                
                S_invivo = cat(2, S_invivo, S_mut);
                T_invivo = cat(2, T_invivo, T_mut);
            else
                % Shuffle and select
                indx = randperm(size(S_invivo, 2));
                S_invivo = S_invivo(:, indx(1:n_invivo));
                T_invivo = T_invivo(:, indx(1:n_invivo));
            end            
        end
        
        dataset_pars.n_invivo = size(S_invivo, 2);
        dataset_pars.n_rand = n_tot - size(S_invivo, 2);
        dataset_pars.add_rand = logical(dataset_pars.n_rand);
        
        if dataset_pars.add_rand
            % Generate random synthetic dataset
            T_lims = [ 0.5     0    0.07*1e-9     0.2*1e-9     -0.46    .03     .03     0; ...
                4      1    1.33*1e-9      4*1e-9       0.86    .30      1      1];
            [S_rand, T_rand, ~, ~, ~] = nsmr_ri_train_data_from_rand(xps_pa, dataset_pars.n_rand, T_lims);
        end        
        
        %% Merge the various training datasets
        
        S_train = cat(2, S_invivo, S_rand);
        T_train = cat(2, T_invivo, T_rand);
        
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
            snr        = unifrnd(snr_min, snr_max, 1, size(T_train, 2));
            S_train    = dd_get_rice_noise( cat(1, S0_train, S2_train), snr);
            
            % Undo Signal normalization
            S_train = bsxfun( @times, S_train, S0_norm_cnst);
            
            % Separate S into S0 and S2
            S0_train = S_train(1:size(S0_train, 1), :);
            S2_train = S_train((size(S0_train, 1) + 1):end, :);
            
        end
        
        % Normalize Signal (for real this time)
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
        [S0_train, ~] = removeconstantrows(S0_train);
        
        % Define fractions of mutated and random datasets relative to the total
        % number of training data points
        % Example -> frac = [frac_invivo frac_rand] = [.2 .8]
        dataset_pars.frac = [size(T_invivo, 2)/size(T_train, 2) size(T_rand, 2)/size(T_train, 2)];
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
    train_fn = ['train_multi_f' ntot_str orig_flag '_' num2str(dataset_pars.frac(1), 2) '_' ...
        num2str(dataset_pars.frac(2), 2) noise_flag '.mat'];
    
    o_path = 'D:\Users\joao\Data\Neural_RI_SMR';
    o_fn = fullfile(o_path, 'Train_data', train_fn);
    save(o_fn, 'train_rep', '-v7.3')
    
end

