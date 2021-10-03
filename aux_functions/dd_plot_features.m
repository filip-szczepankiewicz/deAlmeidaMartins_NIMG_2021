function [axh, ph] = dd_plot_features(feat, target, pn, x)

n_samp = min([size(feat,2) 100000]);
if nargin < 4
    x = 1:n_samp;
else
    x = x(1:n_samp);
end

as.nr = ceil(sqrt( size(feat, 1) ));
as.nc = ceil(size(feat, 1) / as.nr);
as.blk_sp = .25 * 1 / max([as.nr as.nc]);

as.t_marg = 0.05;

as.w_sc = .9;
as.h_sc = .9;

% Define axis limits
dif = max(feat(:)) - min(feat(:));
YLim(1) = min(feat(:)) - .1 * dif;
YLim(2) = max(feat(:)) + .1 * dif;
if exist('target','var')
    if target < min( feat(:) )
        dif = YLim(2) - target;
        YLim(1) = target - .1 * dif;
        YLim(2) = max( feat(:) ) + .1 * dif;
    elseif target > max( feat(:) )
        dif = target - YLim(1);
        YLim(1) = min( feat(:) ) - .1 * dif;
        YLim(2) = target + .1 * dif;
    end
end


axh = []; ph = [];
for i = 1:size(feat, 1)
    
    y = feat(i, 1:n_samp);
    
    ax = jm_sub_axh(as, i, 'cols');
    p1 = plot( x, y, 'r-');
    
    p2 = [];
    if exist('target','var')
        hold on
        p2 = plot(max(x) * [-.1 1.01], target * [1 1], 'k--');
        hold off
    end
        
    ylim(YLim)
        
    if exist('pn', 'var')
        title([pn{i} ' min diff = ' num2str( min( abs(y(:) - target)) )])
    else
        title(['min diff = ' num2str( min( abs(y(:) - target)) )])
    end
      
    axh = cat(1, axh, ax);
    ph = cat(1, ph, [p1 p2]);
end