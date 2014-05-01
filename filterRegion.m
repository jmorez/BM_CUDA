function result=filterRegion(img,i,j,region_size,searchwindow_size,blocksize,percentile)
    
    [rows,cols]=size(img);
    
    padding=(blocksize-1)/2;
    swpadding=floor((searchwindow_size-1)/2);
    
    %Pad the image and restrict the region to/within the unpadded image.
    img_padded=padarray(img,[padding padding]);
    
    min_i=max(1+padding,i+padding);
    max_i=min(rows-padding,i+region_size-1-padding);
    min_j=max(1+padding,j+padding);
    max_j=min(cols-padding,j+region_size-1-padding);
    
    region=img(min_i:max_i,min_j:max_j);
    
    %This will be removed probably
    
    close all
    figure
    imagesc(img_padded);
    rectangle('Position',[min_j min_i (max_i-min_i) (max_j-min_j)],'EraseMode','xor')
    axis image
    colormap('gray')
    

    tic
    %We only need the centroids for the ROI
    [Cx,Cy]=FastCentroid(double(region),blocksize);
    Cx=single(Cx);
    Cy=single(Cy);
    disp(strcat(['Centroids calculated in ' num2str(round(1000*toc)) 'ms. ']))
    mask=circularmask(blocksize);
    
    %Initialize output array containing the matches for each pixel
    result=cell(max_i-min_i,max_j-min_j);

    %Time estimation variables.
    max_iter=(max_i-min_i)*(max_j-min_j);
    counter=0;
    estimate_sample_size=50;
    time_estimate_array=zeros(estimate_sample_size,1);
    
    disp('Launching kernel...')
    for m=min_i:max_i;
        for n=min_j:max_j
            tic
            searchwindow=img((m-swpadding):(m+swpadding),(n-swpadding):(n+swpadding));
            Cx_sw=Cx((m-swpadding):(m+swpadding),(n-swpadding):(n+swpadding));
            Cy_sw=Cy((m-swpadding):(m+swpadding),(n-swpadding):(n+swpadding));
            
            ref=img((m-padding):(m+padding),(n-padding):(n+padding));
            similarity=findMatches(Cx_sw,Cy_sw,ref,Cx(m-i+1,n-j+1),Cy(m-i,n-j),searchwindow,mask);
            result{m-i+1,n-j+1}=selectMatches(region,similarity,percentile);
            %result(m-i+1,n-j+1)=median(matches(:));
            
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