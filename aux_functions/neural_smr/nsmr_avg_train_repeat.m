function net_sel = nsmr_avg_train_repeat(net_coll, n_rep_array, s_list)
% function net_sel = nsmr_avg_train_repeat(net_coll, n_rep_array, s_list)

net_sel = cell(1, numel(n_rep_array));

for s = 1:numel(s_list)
    
    coll_ind = 1;
    s_el     = s_list{s};
    f_names  = fieldnames(net_coll{coll_ind}.(s_el));
    
    for i = 1:numel(n_rep_array)
        n_rep          = n_rep_array(i);
        %
        args      = [f_names cell(size(f_names))]';
        perf_temp = struct(args{:});
        %
        for j = 1:n_rep
            for k = 1:numel(f_names)
                perf_temp.(f_names{k}) = cat(3, perf_temp.(f_names{k}), net_coll{coll_ind}.(s_el).(f_names{k}));
            end
            coll_ind = coll_ind + 1;
        end
        net_sel{i}.([s_el '_av']) = structfun(@(x) median(x, 3), perf_temp, 'UniformOutput', false);
    end
end