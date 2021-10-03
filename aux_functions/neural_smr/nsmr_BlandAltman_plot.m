function axh = nsmr_BlandAltman_plot(x,y,axh,opt)

if (nargin < 3), axh = gca; end
if (nargin < 4), opt = []; end

opt = mplot_opt(opt);

mean_xy = (x + y) / 2;
diff_xy = (x - y);
mean_diff = mean(diff_xy);
std_diff = std(x-y);

hold(axh,'on')

ph1 = scatter(axh,mean_xy,diff_xy,opt.mplot.ms,'filled');
set(ph1,'MarkerFaceAlpha',.01,'MarkerEdgeColor','none','MarkerFaceColor',[0 0 0]);

ph2 = plot(max(mean_xy(:))*[-2 2],mean_diff*[1 1],'-');
ph3 = plot(max(mean_xy(:))*[-2 2],std_diff*[1.96 1.96],'--');
ph4 = plot(max(mean_xy(:))*[-2 2],std_diff*[-1.96 -1.96],'--');

set([ph2 ph3 ph4], 'LineWidth', opt.mplot.lw, 'Color', .7*[1 1 1])

max_x = 1.05*max(mean_xy(:)); min_x = -.05*max(mean_xy(:));
max_y = 1.1*max(diff_xy(:)); min_y = min(diff_xy(:)) - .1*max(diff_xy(:));

set(axh,'Xlim',[min_x max_x],'YLim',[min_y max_y]);

