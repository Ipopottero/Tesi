function [new_train_data, new_test_data] = tsne_scattered(train_data, test_data)

 numComponents = numComponentsFromDataset(train_data);
    
    new_train_data = tsne(train_data, 'NumDimensions', numComponents );
    
    new_test_data = tsne(test_data, 'NumDimensions', numComponents );    
end