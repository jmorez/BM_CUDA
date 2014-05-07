function [Cx,Cy]=FastCentroid(X,blocksize)
%Calculate the centroids for each pixel in an image, scaling as
%O(blocksize*length(X)). 
[rows,cols]=size(X);
    if mod(blocksize,2)~=0 && rows == cols
        Cx=zeros(rows,cols);
        Cy=Cx;
        pad_size=floor((blocksize-1)/2);
        Xpadded=padarray(X,[pad_size pad_size]);
        x=linspace(-blocksize/2,blocksize/2,blocksize);

        for i=1:cols;
            Cx(1:rows,i)=Xpadded((pad_size+1):(rows+pad_size),i:(i+blocksize-1))*x';
            Cy(i,1:rows)=x*Xpadded(i:(i+blocksize-1),(pad_size+1):(rows+pad_size));
        end
        %Normalize and set areas without a centroid to zero to avoid NaN's.
        N=sqrt(Cx.^2+Cy.^2);
        zero=(N==0);
        nonzero=not(zero);
        Cx(nonzero)=Cx(nonzero)./N(nonzero);
        Cy(nonzero)=Cy(nonzero)./N(nonzero);
        Cx(zero)=1;
        Cy(zero)=0;
            
    else
        disp('Block size is not odd or X is not square!')
        Cx=[];
        Cy=[];
    end
end