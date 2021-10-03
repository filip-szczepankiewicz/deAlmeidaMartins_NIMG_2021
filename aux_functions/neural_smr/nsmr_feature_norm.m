function [F_norm, np] = nsmr_feature_norm(F, method, np)
% function [F_norm, np] = nsmr_feature_norm(F, method, np)

if (nargin < 2), method = 'min_max'; end
np.method = method;

switch method
    
    case 'min_max'
        
        if (nargin < 3)
            np.min_norm = min(F,[],2);
            np.max_norm = max(F,[],2);
        end
        
        F_norm = bsxfun(@rdivide, bsxfun(@minus, F, np.min_norm), (np.max_norm - np.min_norm));
        
    case 'mean_norm'
        
        if (nargin < 3)
            np.min_norm = min(F,[],2);
            np.max_norm = max(F,[],2);
            np.mean_norm = mean(F,2);
        end
        
        F_norm = bsxfun(@rdivide, bsxfun(@minus, F, np.mean_norm), (np.max_norm - np.min_norm));
        
    case 'simple_max'
        
        if (nargin < 3)
            np.max_norm = max(F,[],2);
        end
        
        F_norm = bsxfun(@rdivide, F, np.max_norm);
        
end
