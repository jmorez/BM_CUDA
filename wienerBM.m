function filtered = wienerBM(img,matches,Nij)
[rows,cols]=size(matches);
filtered=zeros(rows,cols);
for i=1:rows
    for j=1:cols
        Xij=mean(matches{i,j});
        SNij=mean((matches{i,j}-Xij).^2);
        if (length(matches{i,j})>=1 && SNij ~=0)
        val=Xij+(SNij-Nij)/(SNij)*(img(i,j)-Xij);
        if (val >= 0 && val <= 1)
        filtered(i,j)=Xij+(SNij-Nij)/(SNij)*(img(i,j)-Xij);
        else
            filtered(i,j)=img(i,j);
        end
        else
            filtered(i,j)=img(i,j);
        end
    end
end

end

