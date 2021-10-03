function axh = dd_plot_target_prediction(targ, pred, pn)

n_samp = min([size(targ,2) 20*1e3]);

rand_indx = randperm(size(targ, 2));
targ      = targ(:, rand_indx);
pred      = pred(:, rand_indx);

as.nr = ceil(sqrt( size(targ,1) ));
as.nc = ceil(size(targ,1) / as.nr);
as.blk_sp = .25 * 1 / max([as.nr as.nc]);

as.t_marg = 0.05;

as.w_sc = 1;
as.h_sc = 1;

% Performance metrics
perf_struct = nsmr_get_perf_metrics( targ, pred);

axh = [];
for i = 1:size(targ,1)
    x = targ(i,1:n_samp);
    y = pred(i,1:n_samp);
    
    ax = jm_sub_axh(as, i, 'cols');
    scatter(x, y, 'o', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', .05, 'SizeData', 4)    
   
    xlim(prctile(x, [.1 99.9]))
    ylim(prctile(y, [.1 99.9]))
    
    if exist('pn', 'var')
        title([pn{i} ' r = ' num2str(perf_struct.r(i), 2)])
    else
        title(['r = ' num2str(perf_struct.r(i), 2)])
    end

    hold on
    x_regr = [-1 1]*5; y_regr = perf_struct.m(i) * x_regr + perf_struct.b(i);
    plot(x_regr, y_regr, 'g-');
    
    plot(x_regr, x_regr, 'w-')
    plot(x_regr, x_regr, 'r--')
      
    axh = cat(1, axh, ax);
end
