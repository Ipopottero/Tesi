%% %%%%%%%%%%%%%%%%%%% FIXED PARAMETER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
candidateAlpha = 10.^(-4:2:4);        candidateBeta = 10.^(-4:2:4);    %%
candidateGamma = 10.^(-2:1.5:2);        candidateLambda1 = 10.^(-3:2); %%
candidateLambda2 = 10.^(-3:2);      k           = 100; %D in paper %%
tol = 0.5;                          nGroups     = 10;  %K in paper %%
maxiter = 500;                      gpMaxiter   = 100;             %%
closedFormFun = @groupAPG1;                                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%notice that hMuL wants labes {-1,1}
if min(unique(train_target(:)))==0
    train_target(find(train_target==0))=-1;
    test_target(find(test_target==0))=-1;
end


%the searched parameters are chosen by ﬁve-fold cross validation on the training
%set.
indices = crossvalind('Kfold',train_target(:,1),5);
REC=0;
clear Absolute_true Coverage res
for a = candidateAlpha
    for b = candidateBeta
        for c=candidateGamma
            for d=candidateLambda1
                for e=candidateLambda2
                    
                    for Nfold=1:5
                        test = (indices == Nfold);
                        train = ~test;
                        
                        par           = struct;                         par.maxiter = maxiter;
                        par.gpMaxiter = gpMaxiter;                      par.nGroups = nGroups;
                        par.tol       = tol;                            par.minimumLossMargin = 0.001;
                        par.k         = k;                              par.bQuiet = 1;
                        par.prox_l21  = @prox_l21;                      par.norm_l21  = @norm_l21;
                        
                        par.lambda1   = d;
                        par.lambda2   = e;
                        par.gamma     = c;
                        
                        
                        [n,L] = size(train_target(train,:));
                        Wini = rand(n,k);       Hini = rand(par.k,L);
                        Wini = bsxfun(@rdivide, Wini, sqrt(sum(Wini.^2, 2)));
                        
                        if nGroups > 1
                            [tmpLabelIdx, par.nGroups] = spectralClustering(train_data(train,:), train_target(train,:)',1,0,nGroups);
                        else
                            tmpLabelIdx = ones(1,L);
                        end
                        
                        par.tmpLabelIdx = tmpLabelIdx;
                        
                        [W,H, ~] = closedFormFun(train_data(train,:), train_target(train,:),Wini, Hini, par);
                        
                        optmParameter                   = struct;   optmParameter.maxiter = maxiter;
                        optmParameter.minimumLossMargin = 0.0001;   optmParameter.bQuiet  = 1;
                        
                        optmParameter.alpha             = a;
                        optmParameter.beta              = b;
                        optmParameter.gamma             = c;
                        
                        Z = LLSF( full(train_data(train,:)), W, optmParameter);
                        test_outputs = (train_data(test,:)*Z*H)';
                        
                        pre_labels=test_outputs>0;
                        res(Nfold,1:10) = Evaluation(pre_labels,test_outputs,train_target(test,:)');
                        % input
                        %   test_targets        - L x num_test data matrix of groundtruth labels
                        %   Pre_Labels          - L x num_test data matrix of predicted labels
                        %   Outputs             - L x num_test data matrix of scores
                        
                    end
                    %Calculate performance
                    if mean(res(:,5))>REC
                        REC=mean(res(:,5))
                        Best=[a b c d e];
                    end
                end
            end
        end
    end
end


par           = struct;                         par.maxiter = maxiter;
par.gpMaxiter = gpMaxiter;                      par.nGroups = nGroups;
par.tol       = tol;                            par.minimumLossMargin = 0.001;
par.k         = k;                              par.bQuiet = 1;
par.prox_l21  = @prox_l21;                      par.norm_l21  = @norm_l21;

par.lambda1   = Best(4);
par.lambda2   = Best(5);
par.gamma     = Best(3);


[n,L] = size(train_target);
Wini = rand(n,k);       Hini = rand(par.k,L);
Wini = bsxfun(@rdivide, Wini, sqrt(sum(Wini.^2, 2)));

if nGroups > 1
    [tmpLabelIdx, par.nGroups] = spectralClustering(train_data, train_target',1,0,nGroups);
else
    tmpLabelIdx = ones(1,L);
end

par.tmpLabelIdx = tmpLabelIdx;

[W,H, ~] = closedFormFun(train_data, train_target,Wini, Hini, par);

optmParameter                   = struct;   optmParameter.maxiter = maxiter;
optmParameter.minimumLossMargin = 0.0001;   optmParameter.bQuiet  = 1;

optmParameter.alpha             = Best(1);
optmParameter.beta              = Best(2);
optmParameter.gamma             = Best(3);

Z = LLSF( full(train_data), W, optmParameter);
test_outputs = (test_data*Z*H)';

pre_labels=test_outputs>0;
% clear res
% res(1:5) = Evaluation(pre_labels,test_outputs,test_target');

save(strcat(URI,'LG_',num2str(datas),'_',num2str(fold),'.mat'),'test_outputs')

