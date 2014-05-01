
disp('Clearing all!')
clear all;

%PARAMETERS
imgname='test.jpg';

blocksize=17;
region_i=122;
region_j=196;
regionsize=100;
searchwindow_size=50;
percentile=0.99;




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
filterRegion(img,region_i,region_j,regionsize,searchwindow_size,blocksize,percentile);


%show result