function ms = nsmr_perf_metrics_thresh(w_thresh, T, P, label)
% function nsmr_perf_metrics_thresh(w_thresh, label, T, P)

for i = 1:size(T, 1)
    
    t     = T(i, :);
    p     = P(i, :);    
    
    if label{i} == 'S'
        ind         = T(1, :) < w_thresh;
    elseif label{i} == 'Z'
        ind         = 1 - T(1, :) < w_thresh;
    elseif label{i} == 'T'
        ind         = T(1, :) < w_thresh & T(4, :).^2 < .16;
    else
        ind         = false(size(t));
    end
    
    ms_temp = nsmr_get_perf_metrics( t(~ind), p(~ind));    
    %
    fields = fieldnames(ms_temp);
    for ii = 1:numel(fields)        
        ms.(fields{ii})(i) = ms_temp.(fields{ii});
    end
    
end