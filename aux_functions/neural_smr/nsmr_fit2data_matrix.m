function S = nsmr_fit2data_matrix(m, f)

sz = size(m);
block_sz = min([sz(2) 50*1e3]);

% Separate the rows of the m matrix in various blocks to avoid the creation
% of excessively large matrices 
m = mat2cell(m,sz(1),[block_sz*ones(1,fix(sz(2)/block_sz)) mod(sz(2),block_sz)]);
% Remove empty cell elements
m = m(~cellfun(@isempty,m));

S = [];
for i = 1:numel(m)
    S = cat(2, S, f(m{i}));
end

