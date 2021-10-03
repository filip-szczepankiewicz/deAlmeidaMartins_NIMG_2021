function [axh, ph] = dd_plot_target( targ, targ_nam)
% function [axh, ph] = dd_plot_target( targ, targ_nam)

n_samp = min([size(targ,2) 100*1e3]);

% Axis properties
as.nr = ceil(sqrt( size(targ, 1) ));
as.nc = ceil(size(targ, 1) / as.nr);
as.blk_sp = .15 * 1 / max([as.nr as.nc]);
as.t_marg = 0.05;
as.l_marg = 0.15;
as.w_sc = .9;
as.h_sc = .9;

axh = []; ph = [];
for i = 1:size(targ, 1)
    
    x = targ(i, 1:n_samp);
    
    ax = jm_sub_axh(as, i, 'cols');
    
    p1 = histogram(x, 'DisplayStyle', 'stairs');
    set(ax,'XLim',[min(x) max(x)])
            
    mean_x = round( mean(x), 2, 'significant');
    std_x = round( std(x), 2, 'significant');
    
    if exist('targ_nam', 'var')
        title([targ_nam{i} ' = ' num2str(mean_x) ' \pm ' num2str(std_x)])
    else
        title([num2str(mean_x) ' \pm ' num2str(std_x)])
    end
      
    axh = cat(1, axh, ax);
    ph = cat(1, ph, p1);
end
