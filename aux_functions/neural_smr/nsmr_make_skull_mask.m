function mask = nsmr_make_skull_mask(nii_fn, I_thresh, smooth_thresh)
% mask = nsmr_make_mask(nii_fn, I_thresh, smooth_thresh)

if nargin < 3
    smooth_thresh = .7;
elseif nargin < 2
    I_thresh = 15;
end

% Load signal data
[I,~]  = mdm_nii_read(nii_fn);

% Load xps
xps_fn = mdm_fn_nii2xps(nii_fn);
xps = mdm_xps_load(xps_fn);

% Select and average signal points from b > 1000 shells
indx_b = xps.b / 1e9 > 1;
I_high_b = I(:, :, :, indx_b);
I_high_b = mean(I_high_b, 4);

% Threshold signal
mask = (I_high_b > I_thresh);

% Fill holes
for i = 1:3
    mask = mio_mask_fill(mask, i);
end

% Erode by three voxels
mask = mio_mask_erode(mask, 3);

% Extract largest blob
mask = mio_mask_keep_largest(mask);
 
% Smoothen and threshold mask
filter = 1*[2 2 2];
mask   = mio_smooth_4d(single(mask), filter);
mask   = mask > smooth_thresh;

% Fill holes
for i = 1:3
    mask = mio_mask_fill(mask, i);
end