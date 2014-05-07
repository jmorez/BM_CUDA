ntests=100;
sample_size=25;
sample=zeros(sample_size,1);
t=zeros(2,ntests);
d=zeros(1,ntests);
for i=1:ntests
    for j=1:sample_size
    tic;
    FastCentroid(img,2*i+1);
    sample(j)=toc; 
    end
    t(2,i)=mean(sample);
    d(1,i)=std(sample);
    t(1,i)=2*i+1;
    i
end
errorbar(t(1,:),1000*t(2,:),1000*d);