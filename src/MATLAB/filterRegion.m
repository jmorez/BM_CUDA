function result=filterRegion(img,i,j,region_size,searchwindow_size,blocksize,percentile)
    %Get image dimensions
    [rows,cols]=size(img);
    swpadding=floor((searchwindow_size-1)/2);  
    padding=floor((blocksize-1)/2);
    %Coordinates of the ROI, make sure they stay within the image. Not
    %completely watertight though (see searchwindow padding etc.)
    min_i=max(1,i);
    max_i=min(rows,i+region_size-1);
    min_j=max(1,j);
    max_j=min(cols,j+region_size-1);
    
    %Not sure if needed
    region=img(min_i:max_i,min_j:max_j);
    
    %This will be removed probably
    
%     close all
%     figure
%     imagesc(img_padded);
%     rectangle('Position',[min_j min_i (max_i-min_i) (max_j-min_j)],'EraseMode','xor')
%     axis image
%     colormap('gray')
%     

    tic
    [Cx,Cy]=FastCentroid(img,blocksize);
    Cx=single(Cx);
    Cy=single(Cy);
    disp(strcat(['Centroids calculated in ' num2str(round(1000*toc)) 'ms. ']))
    mask=circularmask(blocksize);
    
    %Initialize output array containing the matches for each pixel
    %result=cell(max_i-min_i,max_j-min_j);
    result=img(min_i:max_i,min_j:max_j);
    %Time estimation variables.
    max_iter=(max_i-min_i)*(max_j-min_j);
    counter=0;
    estimate_sample_size=50;
    time_estimate_array=zeros(estimate_sample_size,1);
    
    %Visualization stuff that will be deleted...
    figure
    hold on;
    imagesc(img);
    axis ij;
    rectangle('Position',[min_j min_i (max_i-min_i) (max_j-min_j)],'EraseMode','xor')
    axis image
    colormap('gray')
    hold off;
    
    %Show realtime filtering
    figure
    h=imagesc(result);
    axis image
    caxis([0 1]);
    colormap('gray');
    drawnow
    disp('Applying kernel to the ROI...')
    for m=min_i:max_i;
        for n=min_j:max_j
            tic
            %Seachwindow coordinates
%             sw_min_i=max(1,m-swpadding);
%             sw_max_i=min(rows,m+searchwindow_size-1);
%             sw_min_j=max(1,n-swpadding);
%             sw_max_j=min(cols,n+searchwindow_size-1);            
            %rectangle('Position',[sw_min_j sw_min_i (sw_max_j-sw_min_j) (sw_max_i-sw_min_i)]);
            %drawnow
            
            %Fetch searchwindow from image, currently ignores any 
            sw_min_i=max(1,m-swpadding);
            sw_max_i=min(rows,m+swpadding);
            sw_min_j=max(1,n-swpadding);
            sw_max_j=min(cols,n+swpadding);
            searchwindow=img(sw_min_i:sw_max_i,sw_min_j:sw_max_j);
            Cx_sw=Cx(sw_min_i:sw_max_i,sw_min_j:sw_max_j);
            Cy_sw=Cy(sw_min_i:sw_max_i,sw_min_j:sw_max_j); 
            ref=img((m-padding):(m+padding),(n-padding):(n+padding));         

            similarity=findMatches(Cx,Cy,ref,Cx(m-i+1,n-j+1),Cy(m-i,n-j),region,mask);
            result{m-i+1,n-j+1}=selectMatches(region,similarity,percentile);

           
            %Find matches for pixel (m,n)
            similarity=findMatches(Cx_sw,Cy_sw,ref,Cx(m,n),Cy(m,n),searchwindow,mask); 
            %This will change probably
            matches=img(similarity<percentile);%selectMatches(region,similarity,percentile);\

            
            mean_est=mean(mean(matches));
            noise_est=mean(mean((mean_est-matches).^2));
            Nij=0.01;
            weight=(noise_est-Nij)/noise_est;
            if(~isempty(matches))
                result(m-i+1,n-j+1)=mean_est+weight*(img(m,n)-mean_est);%length(matches);%
            else
                %result(m-i+1,n-j+1)=img(m-i+1,n-j+1);
            end
            set(h,'CData',result);
            drawnow
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
    hold off;
end
