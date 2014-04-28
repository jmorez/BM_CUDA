function matches=selectMatches(img,similarity,percentile)
    [counts,x]=hist(similarity,linspace(0,1,100));
    treshold=getTreshold(counts,x,percentile);
    k=find(similarity > treshold);
    matches=img(k);
end