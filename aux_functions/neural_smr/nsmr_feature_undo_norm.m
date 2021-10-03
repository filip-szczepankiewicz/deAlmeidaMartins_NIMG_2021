function [F, np] = nsmr_feature_undo_norm(F_norm, np)
% function [F, np] = nsmr_feature_undo_norm(F_norm, np)

switch np.method
    
    case 'min_max'
        
        F = bsxfun(@plus, bsxfun(@times, F_norm, (np.max_norm - np.min_norm)), np.min_norm);
        
    case 'mean_norm'
        
        F = bsxfun(@plus, bsxfun(@times, F_norm, (np.max_norm - np.min_norm)), np.mean_norm);
        
    case 'simple_max'
        
        F = bsxfun(@times, F_norm, np.max_norm);       
end
