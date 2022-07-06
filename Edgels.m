%%% Q3 - Edge Detection %%%
clear all; close all; clc;

%% a: loading images and calculating DoG image:
% Loading image (.jpg) files from subdirectory:
% DoG calculations
figure();

im = imread('C:\Users\user\Downloads\17455098_1317991751623569_1974266347_o.jpg');
DoG = calculateDoG(im);
imagesc(DoG); colormap gray; 
axis off;


%% b+c: Find location of 2 zero crossings in every group of 4 adjecent pixels:

% Create filters of relevant sign changes: 
% twoCrossing1 = [1 1 ; -1 -1];
% twoCrossing2 = [-1 -1; 1 1];
% twoCrossing3 = [1 -1; 1 -1];
% twoCrossing4 = [-1 1; -1 1];

% Looking at the DoG sign image:
imSign = sign(DoG); 
[h,w] = size(imSign);
% Transform into colomns of every 2*2 block in the image:
imDoGFours = double(im2col(DoG,[2,2],'sliding'));
imSignFours = sign(imDoGFours);
% if the signs are different - adding them will give zero.
diff12 = imSignFours(1,:)+imSignFours(2,:);
diff34 = imSignFours(3,:)+imSignFours(4,:);
diff13 = imSignFours(1,:)+imSignFours(3,:);
diff24 = imSignFours(2,:)+imSignFours(4,:);
    
verticals = zeros(1,size(imSignFours,2));
verticals(diff12==0 & diff34==0 & diff13 ~=0 & diff24 ~=0) = 1; % vertical crossing
horizontals = zeros(1,size(imSignFours,2));
horizontals(diff13==0 & diff24==0 & diff12~=0 & diff34~=0) = 1; % horizontal crossing
imEdgels = zeros(1,size(imSignFours,2));
imEdgels(1,verticals==1) = 0.5*((imDoGFours(1,verticals==1)-imDoGFours(2,verticals==1))+...
                                (imDoGFours(3,verticals==1)-imDoGFours(4,verticals==1)));
imEdgels(1,horizontals==1) = 0.5*((imDoGFours(1,horizontals==1)-imDoGFours(3,horizontals==1))+...
                                (imDoGFours(2,horizontals==1)-imDoGFours(4,horizontals==1)));
medEdgel = median(abs(imEdgels(imEdgels~=0)));  
avgEdgel = mean(abs(imEdgels(imEdgels~=0)));
imEdgels = reshape(imEdgels,h-1,w-1);
figure(); subplot(1,2,1); imshow(imEdgels); impixelinfo;
title('Edgel image'); hold on;
    % figure(); imagesc(imEdgels); colormap gray; impixelinfo;                           
    % Thresholding
thresh1 = avgEdgel;
thresh2 = medEdgel;
alpha = 0.25;
thresh = alpha*thresh1+(1-alpha)*thresh2;%1.5*thresh2;
%     imEdgels1 = ones(size(imEdgels));
%     imEdgels1(abs(imEdgels)<thresh1) = 0;
%     imEdgels2 = ones(size(imEdgels));
%     imEdgels2(abs(imEdgels)<thresh2) = 0;
imEdgels3 = ones(size(imEdgels));
imEdgels3(abs(imEdgels)<thresh) = 0;
%     figure();
%     subplot(1,2,1); imshow(imEdgels1); 
%     title('Edgel Image After Thresholding with Average Value');
%     subplot(1,2,2); imshow(imEdgels2); 
%     title('Edgel Image After Thresholding with Median Value');
subplot(1,2,2); imshow(imEdgels3); 
title('Edgel Image After Thresholding');
    
%     figure(); imagesc(imEdgels2); colormap gray;
%     imEdgelsConnect1 = bwmorph(imEdgels1,'bridge',10);
%     imEdgelsConnect2 = bwmorph(imEdgels2,'bridge',10);
imEdgelsConnect3 = bwmorph(imEdgels3,'bridge',10);    
%     figure();
%     subplot(1,2,1); imshow(imEdgelsConnect1); 
%     title('Connected Edgel Image, Average Value Thresholding');
%     subplot(1,2,2); imshow(imEdgelsConnect2); 
%     title('Connected Edgel Image, Median Value Thresholding');
figure(); imshow(imEdgelsConnect3); 
title('Connected Edgel Image');
    