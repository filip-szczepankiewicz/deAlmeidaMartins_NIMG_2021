clear; close all;

f_v = [0:.05:.45 .48]; n_tot_str = '_200';
data_file = cell(1, length(f_v));
net_name = cell(1, length(f_v));
for i = 1:numel(f_v)
    f = f_v(i);
    data_file{1, i} = ['train_RI' n_tot_str '_' num2str(f) '_0_' num2str(1-f) '_noisy.mat'];
    net_name{1, i} = ['net_RI' n_tot_str '_' num2str(f) '_0_' num2str(1-f) '_noisy.mat'];
end

% Select Train and Validation ratios
train_ratio_vec = .85; %logspace(log10(.01),log10(.8),10); %.2:.05:.8; 
val_ratio_vec   = .15 * ones(size(train_ratio_vec));
Ntrials = length(train_ratio_vec);

only_s0 = false;
T_ind = 2:8;
feat_norm_method = 'min_max';
net_design = [150 80 55]; %[150 80 55], [40 20], [70 50 35], [250 180 110]

for j = 1:numel(data_file)
    
    clear net
    % Create network
    net = fitnet(net_design);
    
    %% Prepare paths and load data
    
    dir_path = 'D:\Users\joao\Data\Neural_RI_SMR\';
    
    % Load training dataset
    i_data_fn = fullfile(dir_path, 'Train_data', data_file{j});
    load(i_data_fn); n_voxels = size(T_train, 2);    
    
    nets = cell(1, Ntrials);
    
    if only_s0
        S_train = S0_train;
    else
        S_train = cat(1, S0_train, S2_train);
    end
    
    for i = 1:Ntrials
        
        % Select voxels for training
        n_train = floor( train_ratio_vec(i) * n_voxels );
        T_train_redx = T_train( T_ind, 1:n_train); S_train_redx = S_train(:, 1:n_train);
        
        % Select voxels for validation
        n_val = floor( val_ratio_vec(i) * n_voxels );
        T_val = T_train( T_ind, (n_voxels - n_val + 1):end); S_val = S_train(:, (n_voxels - n_val + 1):end);
        
        %% Feature normalization
        
        [T_train_redx, feat_norm_pars] = nsmr_feature_norm(T_train_redx, feat_norm_method);
        [T_val, ~] = nsmr_feature_norm(T_val, feat_norm_method, feat_norm_pars);
        
        % Check target features
        figure(1), clf
        dd_plot_target( T_train_redx, dataset_pars.T_name(T_ind) );
        
        %% Train network
        
        % Tweak the network (black magic)
        nets{i} = net;
        nets{i}.trainFcn = 'trainscg'; % trainscg
        nets{i}.trainParam.epochs = 15*1e3;
        nets{i}.trainParam.max_fail = 5;
        nets{i}.divideParam.valRatio = .20;
        nets{i}.divideParam.testRatio = .15;
        nets{i}.divideParam.trainRatio = .65;
        
        % User parameters
        nets{i}.userdata.dataset_pars = dataset_pars; % store training data creation parameters
        nets{i}.userdata.xps = xps; %store the xps
        nets{i}.userdata.T_ind = T_ind; %store index of relevant features
        nets{i}.userdata.feat_norm_pars = feat_norm_pars; % store feature normalization parameters
        nets{i}.userdata.train_val_ratios = [train_ratio_vec(i) val_ratio_vec(i)];
        nets{i}.userdata.train_val_nvoxels = [n_train n_val];
        
        % Train the network by showing it the signal and the truth
        nets{i} = train(nets{i}, S_train_redx, T_train_redx, ...
            'useParallel', 'yes', 'useGPU', 'only', 'showResources', 'yes');
        
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
    
    net_fn = fullfile(dir_path,'Train_Networks',net_name{j});
    save(net_fn, 'nets')
end
