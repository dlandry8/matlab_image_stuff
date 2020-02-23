%--------------------------------------------------------------------------
% Function:     Main
% Description:  Calls get_vertical, get_horizontal, detect_edge, 
%                   scale_image_simple, scale_image_averages.
% Variables:    imageMatrix is the grayscale version of the image.
%               sizeImage is the size of the image: [rows, columns]
%               verticalEdge is the vertical edge gradient.
%               sizeVertical is the size of verticalEdge: [rows, columns]
%               magVertical is the magnitude of the verticalEdge
%               horizontalEdge, sizeHorizontal, and magHorizontal are
%                   the same, but for horizontal edge gradient.
%               edgeDetection is the overall edge gradient
%               scale3x is the image scaled down by a factor of 3 without
%                   any filtering.
%               scale4x is the image scaled down by a factor of 4 without
%                   any filtering.
%               scale3x_b is the 3x scale down with a rolling average
%                   filter.
%               scale4x_b is the 4x scale down with rolling avg filter.
%               rowFlip, colFlip, and bothFlip are flipped image
%                   matrices.
%               scaledImage is an upscaled image matrix with regression
%                   to fill in the gaps.
%--------------------------------------------------------------------------
function main
% Obtain the image and convert it to grayscale.
imageMatrix = rgb2gray(imread('DailyShow', 'jpeg'));

% Record the size of the image.
sizeImage = size(imageMatrix);

% Apply the vertical-edge gradient.
verticalEdge = get_vertical(imageMatrix);
sizeVertical = size(verticalEdge);
magVertical = abs(verticalEdge);

% And apply the horizontal-edge gradient.
horizontalEdge = get_horizontal(imageMatrix);
sizeHorizontal = size(horizontalEdge);
magHorizontal = abs(horizontalEdge);

% Use the magnitudes to detect the edges of the image.
edgeDetection = detect_edge(magVertical, magHorizontal);

% Apply 3x and 4x downsampling of the image.
scale3x = scale_image_simple(imageMatrix, 3);
scale4x = scale_image_simple(imageMatrix, 4);

% Apply downsampling with averages of pixels in sample vicinity.
scale3x_b = scale_image_averages(imageMatrix, 3);
scale4x_b = scale_image_averages(imageMatrix, 4);

% In the 3rd part of this assignment, we are given 3 matrices:
%   x[N-n+1, m]
%   x[n, M-m+1]
%   x[N-n+1, M-m+1]
% These flip the image matrix by rows, by columns, and by both,
% respectively. The row_flip and col_flip functions do this to verify our
% assertion.
rowFlip = flipud(imageMatrix);
colFlip = fliplr(imageMatrix);
bothFlip = fliplr(rowFlip);

% Finally, we will upsample the image, interpolating the extra data points
% using bilinear regression.
scaledImage = upsample(imageMatrix);

% Display figure 1: the original grayscale image and the edge-detected
% image.
figure(1);
subplot(2,1,1);
imshow(imageMatrix);
title('Grayscale of Included Image');

subplot(2,1,2);
imshow(cast(edgeDetection, class(imageMatrix)));
title('Edge-Detected image');

% Display figure 2: the edge detection plots.
figure(2);
plotEdges(magVertical, magHorizontal, edgeDetection);

% Display figure 3: the 3x & 4x downsampling by nth sample and by nth
% average sample.
figure(3);
subplot(2,2,1);
imshow(scale3x);
title('Scale by 3 using simple approach');

subplot(2,2,2);
imshow(scale4x);
title('Scale by 4 using simple approach');

subplot(2,2,3);
imshow(scale3x_b);
title('scale by 3 using averages');

subplot(2,2,4);
imshow(scale4x_b);
title('scale by 4 using averages');

% Display the flipped image.
figure(4);
subplot(2,2,1);
imshow(imageMatrix);
title('Original Image');
subplot(2,2,2);
imshow(rowFlip);
title('Flipped image vertically by row');
subplot(2,2,3);
imshow(colFlip);
title('Flipped image horizontally by column');
subplot(2,2,4);
imshow(bothFlip);
title('Flipped image horizontally and vertically');

% Display the upsampled image.
figure(5);
subplot(2,1,1);
imshow(imageMatrix);
title('Original image');
subplot(2,1,2);
imshow(scaledImage);
title('Upsampled Interpolated image');
end

function verticalEdge = get_vertical(imageMatrix)
verticalEdgeMask = [-1 0 1; -2 0 2; -1 0 1];
verticalEdge = conv2(imageMatrix, verticalEdgeMask);
end

function horizontalEdge = get_horizontal(imageMatrix)
horizontalEdgeMask = [1 2 1; 0 0 0; -1 -2 -1];
horizontalEdge = conv2(imageMatrix, horizontalEdgeMask);
end

function edgeDetection = detect_edge(magVert, magHoriz)
edgeDetection = sqrt(magVert.^2 + magHoriz.^2);
end

%--------------------------------------------------------------------------
% Function:     scale_image_simple
% Description:  Scale the image by selecting each (factor)th pixel,
%                   starting with pixel # (ceiling of factor / 2).
%--------------------------------------------------------------------------
function scaledImage = scale_image_simple(imageMatrix, factor)
startingPoint = ceil(factor / 2);
scaledImage = imageMatrix(startingPoint:factor:end, ...
    startingPoint:factor:end);
end


%--------------------------------------------------------------------------
% Function:     scale_image_averages
% Description:  Scale the image by a factor of (factor) by taking the
%                   average value of the selected pixel and those in its
%                   vicinity. It does this by interposing a
%                   (factor x factor) matrix over the original,
%                   averaging the pixel values within that vector, and
%                   copying that value to a set of coordinates of the new
%                   scaled image.
%--------------------------------------------------------------------------
function scaledImage = scale_image_averages(imageMatrix, factor)

% We need a (factor x factor) sub-matrix of the original matrix.
% Initialize the scaled image matrix to a zero matrix to make the program
%   run more efficiently.
sizeOfMatrix = size(imageMatrix);
scaledImage = zeros(uint8(sizeOfMatrix(1)/factor), ...
    uint8(sizeOfMatrix(2)/factor));

% Counters for row & column number for the new scaled matrix.
scaledRow = 1;
scaledCol = 1;

for i = 1:factor:sizeOfMatrix(1)-factor
    for j = 1:factor:sizeOfMatrix(2)-factor
        % temp is our interposed (factor x factor) matrix over the original
        %   image matrix.
        temp = zeros(factor, factor);
        for k = 0:factor-1
            for l = 0:factor-1
                temp(k+1,l+1) = imageMatrix(i+k, j+l);
            end
        end
        scaledImage(scaledRow, scaledCol) = mean2(temp);
        scaledCol = scaledCol + 1;
    end
    scaledCol = 1;
    scaledRow = scaledRow + 1;
end
% Convert the scaled image to uint8. Otherwise, it will not display
% properly.
scaledImage = uint8(scaledImage);
end


function scaledMatrix = upsample(imageMatrix)
% Get the data type of our original image matrix (i.e., uint8)
dataType = class(imageMatrix);
scaledMatrix = interp2(double(imageMatrix));
scaledMatrix = cast(scaledMatrix, dataType);
end
