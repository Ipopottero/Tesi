
%notice that hMuL wants labes {-1,1}
if min(unique(train_target(:)))==0
    train_target(find(train_target==0))=-1;
    test_target(find(test_target==0))=-1;
end

for app=1:2
    
    if app==1
        train_data=train_data1';
        test_data=test_data1';
    elseif app==2
        train_data=train_data2';
        test_data=test_data2';
    elseif app==3
        train_data=train_data3';
        test_data=test_data3';
    end
    
    %the searched parameters are chosen by ﬁve-fold cross validation on the training
    %set.
    indices = crossvalind('Kfold',train_target(:,1),5);
    REC=0;
    clear Absolute_true Coverage res
    for r = 1:2;    % order of Minkowski distance
        for k = 3:3:30;   % number of nearest neighbors
            
            for Nfold=1:5
                test = (indices == Nfold);
                train = ~test;
                
                % overlapping mode
                try
                    W2 = MLDWkNN_train( train_data(train,:),train_target(train,:)',r,k);
                    [~, test_outputs] = MLDWkNN_Pred(train_data(train,:),train_target(train,:)', train_data(test,:), r, k, W2);
                catch
                    W2 = MLDWkNN_train( train_data(train,:),train_target(train,:),r,k);
                    [~, test_outputs] = MLDWkNN_Pred(train_data(train,:),train_target(train,:), train_data(test,:), r, k, W2);
                end
                pre_labels=test_outputs>0;
                res(Nfold,1:10) = Evaluation(pre_labels',test_outputs',train_target(test,:)');
                % input
                %   test_targets        - L x num_test data matrix of groundtruth labels
                %   Pre_Labels          - L x num_test data matrix of predicted labels
                %   Outputs             - L x num_test data matrix of scores
                
            end
            %Calculate performance
            if mean(res(:,5))>REC
                REC=mean(res(:,5))
                Best_r=r;
                Best_k=k;
            end
        end
    end
    
    W2 = MLDWkNN_train(train_data, train_target, Best_r, Best_k);
    [~, test_outputs] = MLDWkNN_Pred(train_data, train_target, test_data, Best_r, Best_k, W2);
    pre_labels=test_outputs>0;
    % res = Evaluation(pre_labels',test_outputs',test_target');
    
    save(strcat(URI,'hML_',num2str(datas),'_',num2str(fold),'_',num2str(app),'_',num2str(itera),'.mat'),'test_outputs')
end

