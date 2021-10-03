clear; close all;

dir_path = 'D:\Users\joao\Data\Neural_SMR\';

train_file = 'train_0.19_0_0.81_noisy.mat'; % Training dataset
net_name = {'net_multi_0.19_0_0.81_noisy.mat'}; % Previously trained network

% Load training dataset
i_data_fn = fullfile(dir_path, 'Train_data', train_file);
load(i_data_fn); n_voxels = size(train_rep{end}.T_train, 2);

for j = 1:numel(net_name)
    
    % Load previously trained network
    net_fn = fullfile(dir_path,'Train_Networks',net_name{j});
    load(net_fn, 'nets'); 
    
    net_orig = nets{1};
    N_nets = numel(nets); clear nets
    
    nets = cell(1, N_nets);
    
    for i = 1:numel(nets)       
        
        T_train = train_rep{i}.T_train;
        S_train = train_rep{i}.S_train;
        
        % User parameters
        T_ind          = net_orig.userdata.T_ind; % index of relevant targets
        T_name         = net_orig.userdata.dataset_pars.T_name; % target names
        train_ratio    = .05; %nets{i}.userdata.train_val_ratios(1);
        val_ratio      = .15; %nets{i}.userdata.train_val_ratios(2);
        feat_norm_pars = net_orig.userdata.feat_norm_pars; % feature normalization parameters          
        
        % Select voxels for training
        n_train      = floor( train_ratio * n_voxels );
        T_train_redx = T_train( T_ind, 1:n_train); S_train_redx = S_train(:, 1:n_train);
        
        % Select voxels for validation
        n_val = floor( val_ratio * n_voxels );
        T_val = T_train( T_ind, (n_voxels - n_val + 1):end); S_val = S_train(:, (n_voxels - n_val + 1):end);
        
        % Feature normalization
        [T_train_redx, ~] = nsmr_feature_norm(T_train_redx, feat_norm_pars.method, feat_norm_pars);
        [T_val, ~]        = nsmr_feature_norm(T_val, feat_norm_pars.method, feat_norm_pars);
        
        %% Check target features
        figure(1), clf
        dd_plot_target( T_train_redx, T_name);        
              
        %% Train the network by showing it the signal input and the targets
        net_orig = train(net_orig, S_train_redx, T_train_redx, ...
            'useParallel', 'yes', 'useGPU', 'only', 'showResources', 'yes');
        
        nets{i} = net_orig;
        
        %% Evaluate Performance
        
        % Performance validation dataset
        P_train_redx = nets{i}(S_train_redx);
        test_train = nsmr_get_perf_metrics( T_train_redx, P_train_redx);
        
        % Performance validation dataset
        P_val = nets{i}(S_val);
        test_val = nsmr_get_perf_metrics( T_val, P_val);
        
        % Show correlation between prediction and target/truth
        figure(2), clf
        dd_plot_target_prediction(T_val, P_val);
        
        % Store performance metrics
        nets{i}.userdata.test_train = test_train;
        nets{i}.userdata.test_val = test_val;
        
    end
    
    net_fn = fullfile(dir_path,'Train_Networks',['reretrain_' net_name{j}]);
    save(net_fn, 'nets')
end
