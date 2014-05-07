
disp('Clearing all!')
clear all;
close all;
%PARAMETERS
imgname='test.jpg';

blocksize=13;
region_i=100;
region_j=100;
regionsize=250;
searchwindow_size=450;
percentile=0.000001;




%Image preparation, resize it to be square
close all
img=imread(imgname);
img=rgb2gray(img);
img=single(img);
img=img/max(max(img));
[rows,cols]=size(img);
newsize=min(rows,cols);
%NOISE
img=img+0*normrnd(0,0.05,rows,cols);
img=img(1:newsize,1:newsize);
%img=(min(min(img))+img)/(max(max(img))-min(min(img)));
%Do stuff
profile on; filterRegion(img,region_i,region_j,regionsize,searchwindow_size,blocksize,percentile); profile off;
profile viewer;
