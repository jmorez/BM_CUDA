%PARAMETERS
imgname='test.jpg';
ref_i=120;
ref_j=227;
%ref_i=274;
%ref_j=323;
blocksize=31;
subplots=0;
treshold=0;

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
disp('Calculating centroids...')
tic
[Cx,Cy]=FastCentroid(img,blocksize);
disp(strcat(['Centroids calculated in: ' num2str(toc) ' seconds.']))
Cx=single(Cx);
Cy=single(Cy);

if(subplots==1)
    subplot(1,2,1)
else
    figure
end
imagesc(img)
colormap('gray')
axis image
title('Search window and reference')
rectangle('Position',[ref_j-padding ref_i-padding blocksize blocksize],'edgecolor','r')

%figure 
%imagesc(ref)
%axis image
%colormap('gray')

disp('Executing kernel...')
tic; C=findMatches(Cx,Cy,ref,Cx(ref_i,ref_j),Cy(ref_i,ref_j),img,mask); 
disp(strcat(['Similarity calculated in: ' num2str(toc) ' seconds.']))
if(subplots==1)
    freezeColors;
    subplot(1,2,2)
else
    figure
end
similarity=1./C((padding+2):(end-padding),(padding+2):(end-padding));
imagesc(similarity.*(similarity > treshold));
%colormap('gray')
colormap('jet')
axis image
title('Similarity')
