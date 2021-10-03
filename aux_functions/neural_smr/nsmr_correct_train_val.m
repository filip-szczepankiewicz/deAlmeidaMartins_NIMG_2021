clear; close all;

f_v = [0:.05:.5 .54 .59 .64 .69 .73 .78 .83 .88 .92 .97]; n_tot_str = '_200';
data_file = cell(1, length(f_v));
net_name = cell(1, length(f_v));
for i = 1:numel(f_v)
    f = f_v(i);
    data_file{1, i} = ['train_multi_f' n_tot_str '_' num2str(f) '_' num2str(1-f) '_noisy'];
    net_name{1, i} = ['net_multi_f' n_tot_str '_' num2str(f) '_' num2str(1-f) '_noisy'];
end

T_ind            = 2:12;
feat_norm_method = 'min_max';

check_perf = true;

%%

for k = 1:numel(data_file)

    % Load training dataset    
    dir_path = 'D:\Users\joao\Data\Neural_SMR\';
    i_data_fn = fullfile(dir_path, 'Train_data', [data_file{k} '.mat']);
    load(i_data_fn);
    
    for j = 1:numel(train_rep)               
        % Load Network
        if numel(train_rep) > 1
            rep_flag = ['_rep_' num2str(j)];
        else
            rep_flag = [];
        end        
        net_fn = fullfile(dir_path, 'Train_Networks', [net_name{k} rep_flag '.mat']);
        load(net_fn);
        
        n_voxels = size(train_rep{j}.T_train, 2);
        
        for i = 1:numel(nets)
            
            net = nets{i};
                                
            % Select Train and Validation ratios
            train_ratio_vec = net.userdata.train_val_ratios(1);
            val_ratio_vec   = net.userdata.train_val_ratios(2);

            % Select voxels for training
            n_train = floor( train_ratio_vec * n_voxels );
            T_train_redx = train_rep{j}.T_train( T_ind, 1:n_train); 
            S_train_redx = train_rep{j}.S_train(:, 1:n_train);
            
            % Select voxels for validation
            n_val = floor( val_ratio_vec * n_voxels );
            T_val = train_rep{j}.T_train( T_ind, (n_voxels - n_val + 1):end); 
            S_val = train_rep{j}.S_train(:, (n_voxels - n_val + 1):end);
            
            %% Feature normalization
            [T_train_norm, ~] = nsmr_feature_norm(T_train_redx, net.userdata.feat_norm_pars.method, net.userdata.feat_norm_pars);
            [T_val_norm, ~]   = nsmr_feature_norm(T_val, net.userdata.feat_norm_pars.method, net.userdata.feat_norm_pars);            
           
            %% Evaluate Performance            
            % Performance training dataset
            P_train_redx      = net(S_train_redx);
            [P_train_redx, ~] = nsmr_feature_undo_norm(P_train_redx, net.userdata.feat_norm_pars);
            test_train        = nsmr_get_perf_metrics( T_train_redx, P_train_redx);
            
            % Performance validation dataset
            P_val      = net(S_val);
            [P_val, ~] = nsmr_feature_undo_norm(P_val, net.userdata.feat_norm_pars);
            test_val   = nsmr_get_perf_metrics( T_val, P_val);            
           
            % Store performance metrics
            nets{i}.userdata.test_train = test_train;
            nets{i}.userdata.test_val   = test_val;                       
        end
        save(net_fn, 'nets')
        
    end
end
