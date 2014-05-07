
disp('Clearing all!')
clear all;

%PARAMETERS
imgname='test.jpg';

blocksize=17;
region_i=122;
region_j=196;
regionsize=100;
searchwindow_size=25;
percentile=0.999999;




%Image preparation, resize it to be square
close all
img=imread(imgname);
img=rgb2gray(img);
img=single(img);
img=img/max(max(img));
[rows,cols]=size(img);
newsize=min(rows,cols);
img=img(1:newsize,1:newsize);

%Do stuff
profile on; filterRegion(img,region_i,region_j,regionsize,searchwindow_size,blocksize,percentile); profile off;
profile viewer;
