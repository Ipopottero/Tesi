function [new_train_data, new_test_data] = kpca_scattered(train_data, test_data)

    % train_data
    
    % use kernel function
    K = rbf_kernel(train_data,train_data, 0.8);
    
    n = size(train_data, 1);
    
    % construct the kernel matrix
    m1 = (1/n)*eye(n);
    m2 = (1/n^2)*ones(n);
    E = (m1-m2)*K;
    
    % find eigenvalues and eigenvectors
    [U, lambda] = eig(E);
    lambda = real(lambda);
    U = real(U);
    
    % sort eigenvalues (descend)
    [~, I]= sort(diag(lambda), 'descend');
    
    U = U(:,I);
    
    % Normalize the eigenvectors
    for i= 1:n
        tmp = U(:,i);
        Up(:,i)= tmp/sqrt(tmp' * K * tmp);
    end
    
    new_train_data = (K - (1/n)*ones(n)*K)*Up;
    
    % test_data
    K = rbf_kernel(test_data,test_data, 0.8);
    n = size(test_data, 1);
    
    m1 = (1/n)*eye(n);
    m2 = (1/n^2)*ones(n);
    E = (m1-m2)*K;
    
    [U, lambda] = eig(E);
    lambda = real(lambda);
    U = real(U);
    
    [~, I]= sort(diag(lambda), 'descend');
    
    U = U(:,I);
    
    for i= 1:n
        tmp = U(:,i);
        Ud(:,i)= tmp/sqrt(tmp' * K * tmp);
    end
    
    new_test_data = (K - (1/n)*ones(n)*K)*Ud;

end