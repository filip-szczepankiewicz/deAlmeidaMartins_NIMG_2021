clear; close all;

do_mask = 1; % Focus on WM

net_name = {'train_RI_1_0_0_noisy.mat'};
n_rand = 50*1e3; % Number of random synthetic datapoints
invivo_dir = '113_FysikDiffusion_183'; % Select in vivo dataset

net_dir = 'D:\Users\joao\Data\Neural_RI_SMR\';

train_cell       = cell(numel(net_name), 1);
train_nvxls_cell = cell(numel(net_name), 1);
val_cell         = cell(numel(net_name), 1);

for j = 1:numel(net_name)
    
    % Load previously trained network
    net_fn = fullfile(net_dir,'Train_Networks',net_name{j});
    load(net_fn)
    n_nets = numel(nets);
    xps = nets{1}.userdata.xps;
    T_ind = nets{1}.userdata.T_ind; n_ind = numel(T_ind);
    T_name = nets{1}.userdata.dataset_pars.T_name;
    
    train.perf = zeros(n_ind, n_nets); val.perf = train.perf;
    train.r = zeros(n_ind, n_nets); val.r = train.r;
    train.m = train.r; val.m = val.r;
    train.b = train.r; val.b = val.r;
    
    train_nvoxels = zeros(n_nets, 1);
    for i = 1:n_nets
        
        % Number of training samples?
        train_nvoxels(i) = nets{i}.userdata.train_val_nvoxels(1);
        
        % Training metrics
        train.perf(:, i) = nets{i}.userdata.test_train.perf;
        train.r(:, i)    = nets{i}.userdata.test_train.r;
        train.m(:, i)    = nets{i}.userdata.test_train.m;
        train.b(:, i)    = nets{i}.userdata.test_train.b;
        
        % Validation metrics
        val.perf(:, i) = nets{i}.userdata.test_val.perf;
        val.r(:, i)    = nets{i}.userdata.test_val.r;
        val.m(:, i)    = nets{i}.userdata.test_val.m;
        val.b(:, i)    = nets{i}.userdata.test_val.b;
        
    end
    
    train_nvxls_cell{j} = train_nvoxels;
    train_cell{j}       = train;
    val_cell{j}         = val;
end

%% Plot risk curves

figure(1), clf
p1 = plot(log10(train_nvxls_cell{1}), sum(train_cell{1}.perf, 1) / n_ind, 'b-');
hold on
p2 = plot(log10(train_nvxls_cell{1}), sum(val_cell{1}.perf, 1) / n_ind, 'r-');

hold off

legend('Train - Big', 'Valid - Big', 'Train', 'Valid', 'Train - Small', 'Valid - Small', 'Train - V Small', 'Valid - V Small')

%% Plot regression metrics

for i = 1:numel(net_name)
    
    figure(2 + 3 * (i - 1)), clf
    [axh, ~] = dd_plot_features(val_cell{i}.r, 1, T_name(T_ind), log10(train_nvxls_cell{i}));
    set(axh, 'XLim', [3.5 6])
    sgtitle('R - train perf')
    
    figure(3 + 3 * (i - 1)), clf
    [axh, ~] = dd_plot_features(val_cell{i}.m, 1, T_name(T_ind), log10(train_nvxls_cell{i}));
    set(axh, 'XLim', [3.5 6])
    sgtitle('m - train perf')
    
    figure(4 + 3 * (i - 1)), clf
    [axh, ~] = dd_plot_features(val_cell{i}.b, 0, T_name(T_ind), log10(train_nvxls_cell{i}));
    set(axh, 'XLim', [3.5 6])
    sgtitle('b - train perf')
    
    figure(5 + 3 * (i - 1)), clf
    [axh, ~] = dd_plot_features(val_cell{i}.perf, 0, T_name(T_ind), log10(train_nvxls_cell{i}));
    set(axh, 'XLim', [3.5 6])
    sgtitle('MSE - train perf')
    
end