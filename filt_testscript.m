filename='test.jpg';
img=imread(filename);
img=rgb2gray(img);
img=single(double(img)/255);
