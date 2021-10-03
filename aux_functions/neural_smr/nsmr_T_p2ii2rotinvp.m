function T_p2 = nsmr_T_p2ii2rotinvp(T_p2ii)
% T_p2 = nsmr_T_p2ii2rotinvp(T_p2ii)
% T_p2ii = [T_p20   T_p21r   T_p21i   T_p22r   T_p22i]

T_p2_temp = cat(3, T_p2ii(1, :), ...
    T_p2ii(2, :) + 1i * T_p2ii(3, :), -T_p2ii(2, :) + 1i * T_p2ii(3, :), ...
    T_p2ii(4, :) + 1i * T_p2ii(5, :), T_p2ii(4, :) - 1i * T_p2ii(5, :));
%
T_p2      = sqrt( sum( abs( T_p2_temp ).^2, 3) ) / sqrt( 5 / (4 * pi));