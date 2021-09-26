function numComponents = numComponentsFromDataset(X)

    [~, score, ~,~,explained,mu]= pca(X);
    sum_explained = 0;
    idx = 0;
    while sum_explained < 99
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    
    numComponents = idx;

end
