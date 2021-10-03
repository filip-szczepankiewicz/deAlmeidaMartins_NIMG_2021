function mask = nsmr_make_mask(nii_fn, mfs_fn, mask_WM, uFA_nii)
% function mask = nsmr_make_mask(nii_fn, mfs_fn, mask_WM, uFA_nii)

if (nargin < 3), mask_WM = 0; end
if (nargin < 4), uFA_nii = []; end

% Load or create skull mask
try
    skull_mask = mdm_nii_read(fullfile(fileparts(nii_fn), 'nsmr_skull_mask.nii.gz'));
    skull_mask = double(skull_mask);
catch
    skull_mask = nsmr_make_skull_mask(nii_fn, 16, .6);
end

% Create s0 thresh mask
mfs     = mdm_mfs_load(mfs_fn); % Load mfs structure
s0      = mfs.m(:,:,:,1);
s0_mask = s0 > .05*max(s0(:));
mask    = s0_mask .* skull_mask;

if mask_WM
    % Create uFA mask
    [I_uFA, ~]  = mdm_nii_read(uFA_nii); % Load uFA maps
    uFA_mask = I_uFA > .6;
    
    mask = uFA_mask .* mask;
end

