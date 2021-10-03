function maps_grid_plot(im2d)

w = ceil(sqrt(size(im2d,3)));
h = ceil(size(im2d,3)/w);

clim = [0 max(im2d(:))];

for n_slc = 1:size(im2d,3)
    
    axh = subplot(h,w,n_slc);
    
    imagesc(im2d(:,:,n_slc)')
    set(axh,'YDir','normal')
    axis(axh,'tight','off')
    set(axh,'CLim',clim)
    colormap(axh,'gray')
end