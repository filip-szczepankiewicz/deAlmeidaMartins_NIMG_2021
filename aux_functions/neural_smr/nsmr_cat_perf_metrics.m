function [train, val, deploy] = nsmr_cat_perf_metrics(nets, S_deploy, T_deploy)
% function [train, val, deploy] = nsmr_cat_perf_metrics(net_fn, S_deploy, T_deploy)

for i = 1:numel(nets)
    % Load previously trained network
    net = nets{i};
    T_ind = net.userdata.T_ind;
    
    % Training metrics
    f_names = fieldnames(net.userdata.test_train);
    if ~exist('train','var')
        args  = [f_names cell(size(f_names))]';
        train = struct(args{:});
    end
    for f = 1:numel(f_names)
        train.(f_names{f}) = cat(2, train.(f_names{f}), net.userdata.test_train.(f_names{f}));
    end
    
    % Validation metrics
    f_names = fieldnames(net.userdata.test_val);
    if ~exist('val','var')
        args = [f_names cell(size(f_names))]';
        val   = struct(args{:});
    end
    for f = 1:numel(f_names)
        val.(f_names{f}) = cat(2, val.(f_names{f}), net.userdata.test_val.(f_names{f}));
    end          
    
    if nargin > 1
        % Performance metrics
        P_deploy  = net(S_deploy); % Predict data
        P_deploy  = nsmr_feature_undo_norm(P_deploy, net.userdata.feat_norm_pars);
        perf_temp = nsmr_get_perf_metrics( T_deploy(T_ind, :), P_deploy);
        %     [T_deploy_norm, ~] = nsmr_feature_norm( T_deploy(T_ind, :), ... % Feature normalization
        %         net.userdata.feat_norm_pars.method, net.userdata.feat_norm_pars);
        %
        f_names = fieldnames(perf_temp);
        if ~exist('deploy','var')
            args   = [f_names cell(size(f_names))]';
            deploy = struct(args{:});
        end
        for f = 1:numel(f_names)
            deploy.(f_names{f}) = cat(2, deploy.(f_names{f}), perf_temp.(f_names{f}));
        end
    else
        deploy = [];
    end
    
  
end