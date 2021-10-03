function [axh,as] = jm_sub_axh(as, a_indx, count_dir)
% function [axh,as] = jm_sub_axh(as, a_indx, count_dir)

if (nargin < 3), count_dir = 'cols'; end
if (nargin < 2), a_indx = 1; end
if (nargin < 1), as = struct(); end


as = msf_ensure_field(as, 'nr', 1);
as = msf_ensure_field(as, 'nc', 1);
as = msf_ensure_field(as, 'blk_sp', 0);

as = msf_ensure_field(as, 'l_marg', .05);
as = msf_ensure_field(as, 'r_marg', as.l_marg);
as = msf_ensure_field(as, 'b_marg', .05);
as = msf_ensure_field(as, 't_marg', as.b_marg);

as.w = ((1 - as.r_marg - as.l_marg) / as.nc) - as.blk_sp;
as.h = ((1 - as.b_marg - as.t_marg) / as.nr) - as.blk_sp;

as = msf_ensure_field(as, 'w_sc', 1);
as = msf_ensure_field(as, 'h_sc', 1);

if strcmp(count_dir,'cols')
    
    c_indx = rem(a_indx - 1, as.nc) + 1;
    r_indx = (a_indx - c_indx)/as.nc + 1;
    
    left = as.l_marg + (as.w + as.blk_sp)*(c_indx-1);
    bottom = (1 - as.t_marg) - (as.h + as.blk_sp)*(r_indx);
    width = as.w_sc*as.w;
    height = as.h_sc*as.h;
    
    axh = axes('position',[left bottom width height]);
    
elseif strcmp(count_dir,'rows')
    
    r_indx = rem(a_indx - 1, as.nr) + 1;
    c_indx = (a_indx - r_indx)/as.nr + 1;
    
    left = as.l_marg + (as.w + as.blk_sp)*(c_indx-1);
    bottom = (1 - as.t_marg) - (as.h + as.blk_sp)*(r_indx);
    width = as.w_sc*as.w;
    height = as.h_sc*as.h;
    
    axh = axes('position',[left bottom width height]);
    
end



