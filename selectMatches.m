function matches=selectMatches(img,similarity,percentile)
    [counts,x]=hist(similarity,linspace(0,1,1000));
    treshold=getTreshold(counts,x,percentile);
    k=find(similarity > treshold);
    matches=img(k);
end