function mask=circularmask(masksize)
%Returns a circular mask with edge <size>
    R=(masksize(1)-1)/2;
    [i,j] = meshgrid(1:masksize(1));
    mask = sqrt((i-R-1).^2+(j-R-1).^2)<R;
end
