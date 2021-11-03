function [test_data, test_target, train_data, train_target] = preprocessing(test_data, test_target, train_data, train_target, use_pca)

    %pca_n_classes non si usa :'(

        % Rimuovo i null
        for i = [size(train_data,2): -1 : 1]
            U = unique(train_data(:,i));
            if size(U)==1
                train_data(:,i)=[];
                test_data(:,i)=[];
            end
        end

        if use_pca     %PCA
            [coeff,scoreTrain,~,~,explained,mu] = pca(train_data);
            sum_explained = 0;
            idx = 0;
            while sum_explained < 99
                idx = idx + 1;
                sum_explained = sum_explained + explained(idx);
            end
            train_data=scoreTrain(:,1:idx);
            test_data= (test_data-mu)*coeff(:,1:idx);
        end
        
        %porto tutti i target in {-1,1}
        train_target(find(train_target==0))=-1;
        test_target(find(test_target==0))=-1;
        

end

