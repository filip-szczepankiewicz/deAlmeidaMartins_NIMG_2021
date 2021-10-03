clear; close all;

data_file = {'train_ri_optimized_182183'};
net_name  = {'net_ri_optimized_182183'};


% Select Train and Validation ratios
train_ratio_vec = .85; %logspace(log10(.01), log10(.85), 20); %.2:.05:.8;
val_ratio_vec   = .15 * ones(size(train_ratio_vec));
Ntrials         = length(train_ratio_vec);

only_s0          = false;
net_design       = [150 80 55]; %[150 80 55], [40 20], [70 50 35], [250 180 110]
T_ind            = 2:8;
feat_norm_method = 'min_max';

check_perf = true;

for k = 1:numel(data_file)
    
    % Create network
    net = fitnet(net_design);
    
    %% Prepare paths and load data
    
    dir_path = pwd;
    
    % Load training dataset
    i_data_fn = fullfile(dir_path, 'Train_data', [data_file{k} '.mat']);
    load(i_data_fn);
    
    for j = 1:numel(train_rep)               
        clear nets
        nets = cell(1, Ntrials);
        
        n_voxels = size(train_rep{j}.T_train, 2);
        
        if only_s0
            train_rep{j}.S_train = train_rep{j}.S0_train;
            s0_flag = '_s0';
        else
            train_rep{j}.S_train = cat(1, train_rep{j}.S0_train, train_rep{j}.S2_train);
            s0_flag = [];
        end
        
        for i = 1:Ntrials
            
            % Select voxels for training
            n_train = floor( train_ratio_vec(i) * n_voxels );
            T_train_redx = double(train_rep{j}.T_train( T_ind, 1:n_train)); 
            S_train_redx = double(train_rep{j}.S_train(:, 1:n_train));
            
            % Select voxels for validation
            n_val = floor( val_ratio_vec(i) * n_voxels );
            T_val = double(train_rep{j}.T_train( T_ind, (n_voxels - n_val + 1):end)); 
            S_val = double(train_rep{j}.S_train(:, (n_voxels - n_val + 1):end));
            
            %% Feature normalization            
            [T_train_redx, feat_norm_pars] = nsmr_feature_norm(T_train_redx, feat_norm_method);
            [T_val, ~]                     = nsmr_feature_norm(T_val, feat_norm_method, feat_norm_pars);
            
            %% Train network
            
            % Tweak the network
            nets{i}                        = net;
            nets{i}.trainFcn               = 'trainscg'; % trainscg
            nets{i}.trainParam.epochs      = 20*1e3;
            nets{i}.trainParam.max_fail    = 5;
            nets{i}.divideParam.valRatio   = .25;
            nets{i}.divideParam.testRatio  = .01;
            nets{i}.divideParam.trainRatio = .74;
            
            % User parameters
            nets{i}.userdata.dataset_pars      = train_rep{j}.dataset_pars; % store training data creation parameters
            nets{i}.userdata.xps               = train_rep{j}.xps; %store the xps
            nets{i}.userdata.xps_pa            = train_rep{j}.xps_pa; %store the PA xps
            nets{i}.userdata.T_ind             = T_ind; %store index of relevant features
            nets{i}.userdata.feat_norm_pars    = feat_norm_pars; % store feature normalization parameters
            nets{i}.userdata.train_val_ratios  = [train_ratio_vec(i) val_ratio_vec(i)];
            nets{i}.userdata.train_val_nvoxels = [n_train n_val];
            
            % Train the network by showing it the signal and the truth
            nets{i} = train(nets{i}, S_train_redx, T_train_redx, ...
                'useParallel', 'yes', 'useGPU', 'only', 'showResources', 'yes');
            
            
            %% Evaluate Performance
            
            % Performance validation dataset
            P_train_redx = nets{i}(S_train_redx);
            test_train   = nsmr_get_perf_metrics( T_train_redx, P_train_redx);
            
            % Performance validation dataset
            P_val    = nets{i}(S_val);
            test_val = nsmr_get_perf_metrics( T_val, P_val);
            
            if check_perf
                % Check target features
                figure(1), clf
                dd_plot_target( T_train_redx, train_rep{j}.dataset_pars.T_name);
                
                % Show correlation between prediction and target/truth
                figure(2), clf
                dd_plot_target_prediction(T_val, P_val);
            end
            
            % Store performance metrics
            nets{i}.userdata.test_train = test_train;
            nets{i}.userdata.test_val   = test_val;            
        end
        
        if numel(train_rep) > 1
            rep_flag = ['_rep_' num2str(j)];
        else
            rep_flag = [];
        end       
       
        net_fn = fullfile(dir_path, 'Trained_Networks', [net_name{k} s0_flag rep_flag '.mat']);
        save(net_fn, 'nets')
        
    end
end
