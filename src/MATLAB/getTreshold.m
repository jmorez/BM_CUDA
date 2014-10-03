function X=getTreshold(counts,x,percentile)
    %Returns the x value for which holds that <percentile> % of the counts
    %are lower than this. Intensity counts are assumed to go from 0
    normalized=counts/sum(counts);
    sizecounts=length(counts);
    i=1;
    if percentile==0
        X=0;
    elseif percentile==1
        X=1;
    else
        integr=0;
        while integr < percentile && i <= sizecounts
            integr=integr+normalized(i);
            i=i+1;
        end
        X=x(i-1);
    end
end
