%PARAMETERS
imgname='test.jpg'
ref_i=157;
ref_j=219;
blocksize=25;

close all
img=imread(imgname);
img=rgb2gray(img);
img=single(img);
img=img/max(max(img));
[rows,cols]=size(img);
newsize=min(rows,cols);
newsize=newsize-1*(mod(newsize+1,2));
img=img(1:newsize,1:newsize);
padding=(blocksize-1)/2;
ref=img((ref_i-padding):(ref_i+padding),(ref_j-padding):(ref_j+padding));
mask=circularmask(blocksize);
[Cx,Cy]=FastCentroid(img,blocksize);
Cx=single(Cx);
Cy=single(Cy);

subplot(1,2,1)
h1=imagesc(img)
colormap('gray')
axis image
title('Search window and reference')
rectangle('Position',[ref_j-padding ref_i-padding blocksize blocksize])
freezeColors;
%figure 
%imagesc(ref)
%axis image
%colormap('gray')

tic; C=findMatches(Cx,Cy,ref,img,mask); toc;
subplot(1,2,2)
h2=imagesc(1./C);
colormap('jet')
axis image
title('Similarity')