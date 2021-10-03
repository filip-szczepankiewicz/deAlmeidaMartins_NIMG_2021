function mask = jm_skull_strip(I, threshold, n_erode, n_dil, check)
% mask = jm_skull_strip(I, threshold, n_erode, n_dil, check)

if (nargin < 3), n_erode = 2; n_dil = n_erode; check = 0; end
if (nargin < 4), n_dil = n_erode; check = 0; end
if (nargin < 5), check = 0; end

fontSize = 12;

%% ===========================================================================================================
% Remove NaN
indx = isnan(I); I(indx) = 0;

% Generate mask
mask = zeros(size(I));

% Get number of slices
[~, ~, N_slc] = size(I);

for n_slc = 1:N_slc
    
    % Threshold the image to make a binary image.
    binaryImage = I(:,:,n_slc) > threshold;
    
    % Extract the two largest blobs, which will either be the skull and brain,
    % or the skull/brain (if they are connected) and small noise blob.
    binaryImage = bwareafilt(binaryImage,2);		% Extract 2 largest blobs.
    
    % Erode it a little with imopen().
    binaryImage = imopen(binaryImage, strel('disk',n_erode));
    
    % Now brain should be disconnected from skull, if it ever was.
    % So extract the brain only - it's the largest blob.
    binaryImage = bwareafilt(binaryImage, 1);		% Extract largest blob.
    
    % Fill any holes in the brain.
    binaryImage = imfill(binaryImage, 'holes');
    
    % Dilate mask out a bit in case we've chopped out a little bit of brain.
    binaryImage = imdilate(binaryImage, strel('disk',n_dil));
    
    %% ===========================================================================================================
    % Check if skull stripping went OK
    if check == 1
        
        I_slc = I(:,:,n_slc);
        
        % Display original image.
        subplot(2, 3, 1);
        imshow(I_slc, []);
        axis on;
        caption = sprintf('Original Grayscale Image\n%s');
        title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
        drawnow;
        hp = impixelinfo();        
       
        % Make the pixel info status line be at the top left of the figure.
        hp.Units = 'Normalized';
        hp.Position = [0.01, 0.97, 0.08, 0.05];
        
        % Display the histogram so we can see what gray level we need to threshold it at.
        subplot(2, 3, 2:3);
        % For MRI images, there is tipically a huge number of black pixels with gray level less than about 1,
        % and that makes a huge spike at the first bin.  Ignore those pixels so we can get a histogram of just non-zero pixels.
        
        [pixelCounts, grayLevels] = histcounts(I_slc(I_slc >= 1), 100);
        
        grayLevels_avg = (grayLevels(1:(numel(grayLevels)-1)) + grayLevels(2:numel(grayLevels)))/2;
        
        faceColor = .7 * [1 1 1]; 
        bar(grayLevels_avg, pixelCounts, 'BarWidth', 1, 'FaceColor', faceColor);
        
        % Find the last gray level and set up the x axis to be that range.
        indx = find(pixelCounts>0, 1, 'last');
        lastGL = grayLevels_avg(indx);
        xlim([0, lastGL]);
        grid on;
        
        % Set up 5 tick marks.
        ax = gca;
        ax.XTick = linspace(0,lastGL,5);
        title('Histogram of Non-Black Pixels', 'FontSize', fontSize, 'Interpreter', 'None', 'Color', faceColor);
        xlabel('Gray Level', 'FontSize', fontSize);
        ylabel('Pixel Counts', 'FontSize', fontSize);
        drawnow;
        
        % Display the tresholded image.
        subplot(2, 3, 4);
        imshow(I_slc > threshold, []);
        axis on;
        caption = sprintf('Initial Binary Image');
        title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
        
        % Display the final binary image.
        subplot(2, 3, 5);
        imshow(binaryImage, []);
        axis on;
        caption = sprintf('Final Binary Image\nof Brain Alone');
        title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
        
        % Mask out the skull from the original gray scale image.
        skullFreeImage = I_slc; % Initialize
        skullFreeImage(~binaryImage) = 0; % Mask out.
        % Display the image.
        subplot(2, 3, 6);
        imshow(skullFreeImage, []);
        axis on;
        caption = sprintf('Gray Scale Image\nwith Skull Stripped Away');
        title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
        
        disp('Press any key to continue')
        pause;
        
    end
    
    
    mask(:,:,n_slc) = binaryImage;
    
end
