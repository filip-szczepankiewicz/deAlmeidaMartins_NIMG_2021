function m_ri = nsmr_m2rotinvm(m)

m_ri = zeros([size(m, 1:3) 8]);
m_ri(:, :, :, 1:7) = m(:, :, :, [1 2 3 4 5 11 12]);

m_ri_8_temp = cat(5, m(:,:,:, 6), ...
    m(:,:,:, 7) + 1i*m(:,:,:, 8), -m(:,:,:, 7) + 1i*m(:,:,:, 8), ...
    m(:,:,:, 9) + 1i*m(:,:,:, 10), m(:,:,:, 9) - 1i*m(:,:,:, 10));
m_ri(:, :, :, 8) = sqrt( sum( abs( m_ri_8_temp ).^2, 5) ) / sqrt( 5 / (4 * pi));
clear m_ri_8_temp

% Detect and replace p2 > 1 solutions
% m_ri_8 = m_ri(:, :, :, 8);
% indx   = m_ri_8 > 1; m_ri_8(indx) = ones(nnz(indx),1);
% %
% m_ri(:, :, :, 8) = m_ri_8;
clear m_ri_8