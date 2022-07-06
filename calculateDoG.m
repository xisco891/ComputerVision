function DoGimage = calculateDoG(im)
% Applies Difference of Gaussian filter on the give image 'im'

im = rgb2gray(im);
% [h,w] = size(im);

% Creating the DoG filter
sigma1 = 1;
sigma2 = 20;
gauss1 = fspecial('gaussian',11,sigma1);
gauss2 = fspecial('gaussian',11,sigma2);
DoG = gauss2-gauss1;

% % Presenting the filters:
% figure();
% subplot(1,3,1); imagesc(gauss1); colormap gray;
% title(['1st Gaussian Filter, \sigma_1=',num2str(sigma1)]); hold on;
% subplot(1,3,2); imagesc(gauss2); colormap gray;
% title(['2nd Gaussian Filter, \sigma_2=',num2str(sigma2)]);
% subplot(1,3,3); imagesc(DoG); colormap gray; 
% title('DoG Filter');

% Convolving image with the DoG filter: 
DoGimage = conv2(double(im),DoG,'same');

end