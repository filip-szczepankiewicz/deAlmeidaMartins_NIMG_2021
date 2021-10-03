function ms = nsmr_get_perf_metrics( targ, pred)

ms.MSE   = sum((targ - pred).^2, 2) ./ size(targ, 2);
ms.NRMSE = sqrt(ms.MSE) ./ ( max(targ, [], 2) - min(targ, [], 2));

[~, ms.m, ms.b] = regression(targ, pred);
%
ms.r            = zeros( size(ms.MSE) );
for i = 1:size(targ,1)
%    rob_fit = robustfit(targ(i, :)', pred(i, :)');
% 
%    ms.b(i) = rob_fit(1);
%    ms.m(i) = rob_fit(2);
   
   r_mat = corrcoef(targ(i, :), pred(i, :));
   ms.r(i) = r_mat(2, 1);
end
