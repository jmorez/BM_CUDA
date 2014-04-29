function result=filterRegion(img,i,j,region_size,blocksize,percentile)
    [rows,cols]=size(img);
    
    %Figure out region
    min_i=max(1,i);
    max_i=min(rows,i+region_size-1);
    min_j=max(1,j);
    max_j=min(cols,j+region_size-1);
    
%     
%     if(mod((max_i-min_i+1),2)==0)
%         max_i=max_i-1
%     elseif(mod((max_j-min_j+1),2)==0)
%         max_j=max_j-1
%     end
    
    region=img(min_i:max_i,min_j:max_j);

    close all
    figure
    imagesc(img);
    rectangle('Position',[min_j min_i (max_i-min_i) (max_j-min_j)],'edgecolor','r')
    axis image
    colormap('gray')
    
    padding=(blocksize-1)/2;
    tic
    [Cx,Cy]=FastCentroid(double(region),blocksize);
    Cx=single(Cx);
    Cy=single(Cy);
    disp(strcat(['Centroids calculated in ' num2str(round(1000*toc)) 'ms. ']))
    mask=circularmask(blocksize);
    result=zeros(max_i-min_i,max_j-min_j);


    %Time estimation variables.
    max_iter=(max_i-min_i)*(max_j-min_j);
    counter=0;
    estimate_sample_size=50;
    time_estimate_array=zeros(estimate_sample_size,1);
    
    disp('Launching kernel...')
    for m=(i+padding):(i+(max_i-min_i)-padding)
        for n=(j+padding):(j+(max_j-min_j)-padding)
            tic
            ref=img((m-padding):(m+padding),(n-padding):(n+padding));
            similarity=findMatches(Cx,Cy,ref,Cx(m,n),Cy(m,n),region,mask);
            matches=selectMatches(region,similarity,percentile);
            result(m-i+1,n-j+1)=median(matches(:));
            
            %More waitbar/time estimation stuff
            timeleft=(max_iter-counter)*toc;
            time_estimate_array(mod(counter,estimate_sample_size)+1)=timeleft;
            mean_time=sum(time_estimate_array(:))/estimate_sample_size;
            if(mod(counter,round(0.1*max_iter/log(max_iter)))==0 && counter > 1)
                hours=floor(mean_time/3600);
                minutes=floor((mean_time-3600*hours)/60);
                seconds=floor((mean_time-3600*hours-60*minutes));
                disp(strcat(...
                    ['Estimated time left: '            ...
                    num2str(hours) 'h ' ...
                    num2str(minutes) 'm '   ...
                    num2str(seconds) 's']));
            end
            counter=counter+1;
        end
    end 
end