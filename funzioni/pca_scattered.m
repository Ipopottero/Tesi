function [new_train_data, new_test_data] = pca_scattered(train_data, test_data)

    [~, score]= pca(train_data, 'numComponents', 166);
    new_train_data= score;
    
    [~, score]= pca(test_data, 'numComponents', 80);
    new_test_data= score;
    
end