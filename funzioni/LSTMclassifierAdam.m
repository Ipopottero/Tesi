% function GRUclassifier(train_data,train_target, test_data,test_target)
%https://it.mathworks.com/help/deeplearning/ug/multilabel-text-classification-using-deep-learning.html
close all force

for reitero=1:3
    if reitero==1
        numEpochs = 20;
    elseif reitero==2
        numEpochs = 50;
    elseif reitero==3
        numEpochs = 75;
    end
    
    
    for id=[2 6 11 26]
        close all force
        
        for itero=1:10
            
            embeddingDimension = 1;
            numHiddenUnits = 250;
            inputSize = size(train_data,2) + 1;
            numClasses=size(train_target,2);
            numObservations=size(train_data,1);
            numFeatures=size(train_data,2);
            
            parameters = struct;
            % parameters.lstm.weights = dlarray(randn(4*numHiddenUnits,1),'CU');
            % parameters.lstm.recurrent = dlarray(randn(4*numHiddenUnits,numHiddenUnits),'CU');
            % parameters.lstm.bias = dlarray(randn(4*numHiddenUnits,1),'C');
            parameters.lstm.weights = dlarray(initializeGlorot(4*numHiddenUnits,1));
            parameters.lstm.recurrent = dlarray(initializeGlorot(4*numHiddenUnits,numHiddenUnits));
            parameters.lstm.bias = dlarray(zeros(4*numHiddenUnits,1,'single'));
            
            parameters.fc.Weights = dlarray(initializeGaussian([numClasses,numHiddenUnits]));
            parameters.fc.Bias = dlarray(zeros(numClasses,1,'single'));
            
            miniBatchSize = 30;
            learnRate = 0.001;
            gradientDecayFactor = 0.15;
            squaredGradientDecayFactor = 0.999;
            
            gradientThreshold = 1;
            plots = "training-progress";
            labelThreshold = 0.5;
            numObservationsTrain = size(train_data,1);
            numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
            validationFrequency = numIterationsPerEpoch;
            
            executionEnvironment = "auto";
            
            if plots == "training-progress"
                figure
                
                % Labeling F-Score.
                subplot(2,1,1)
                lineLossTrain = animatedline('Color',[0 0.447 0.741]);
                
                ylim([0 1])
                xlabel("Iteration")
                ylabel("Labeling F-Score")
                grid on
                
            end
            
            trailingAvg = [];
            trailingAvgSq = [];
            
            iteration = 0;
            start = tic;
            i=1;
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
            
            % Read mini-batch of data and convert the labels to dummy
            % variables.
            X = train_data(idx,:);
            % parameters.lstm.H0 = dlarray(randn(numHiddenUnits,size(X,1)),'CB');
            % parameters.lstm.C0 = dlarray(randn(numHiddenUnits,size(X,1)),'CB');
            
            % Loop over epochs.
            for epoch = 1:numEpochs
                
                % Loop over mini-batches.
                for i = 1:numIterationsPerEpoch
                    iteration = iteration + 1;
                    idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
                    
                    % Read mini-batch of data and convert the labels to dummy
                    % variables.
                    X = train_data(idx,:);
                    parameters.lstm.H0 = dlarray(randn(numHiddenUnits,size(X,1)),'CB');
                    parameters.lstm.C0 = dlarray(randn(numHiddenUnits,size(X,1)),'CB');
                    
                    %labels (1,0), numClasses*miniBatchSize
                    labels = train_target(idx,:)';
                    if min(train_target(:))==-1
                        labels(find(labels==-1))=0;
                    end
                    
                    % Convert mini-batch of data to dlarray.
                    dlX = dlarray(X,'BTC');
                    
                    % If training on a GPU, then convert data to gpuArray.
                    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                        dlX = gpuArray(dlX);
                    end
                    
                    % Evaluate the model gradients, state, and loss using dlfeval and the
                    % modelGradients function.
                    [gradients,loss,dlYPred] = dlfeval(@modelGradientsLSTM, dlX, labels, parameters);
                    
                    
                    % Gradient clipping.
                    if iteration>1
                        gradOld=gradients;
                    end
                    gradients = dlupdate(@(g) thresholdL2Norm(g, gradientThreshold),gradients);
                    if iteration==1
                        gradOld=gradients;
                    end
                    
                    % Update the network parameters using the Adam optimizer.
                    n_iter = numEpochs * numIterationsPerEpoch;
                    [parameters,trailingAvg,trailingAvgSq] = adamupdateStoc(parameters,gradients, ...
                        trailingAvg,trailingAvgSq,iteration,n_iter,learnRate,gradientDecayFactor,squaredGradientDecayFactor, epsilon,id-1,gradOld);
                    
                    
                    % Display the training progress.
                    score = labelingFScore(extractdata(dlYPred) > labelThreshold,labels);
                    
                    addpoints(lineLossTrain,iteration,double(gather(score)))
                    
                    drawnow
                    
                end
                
                % Shuffle data.
                idx = randperm(numObservationsTrain);
                train_data = train_data(idx,:);
                train_target = train_target(idx,:);
            end
            
            %test set
            dlX = dlarray(test_data,'BTC');
            parameters.lstm.H0 = dlarray(randn(numHiddenUnits,size(dlX,2)),'CB');
            parameters.lstm.C0 = dlarray(randn(numHiddenUnits,size(dlX,2)),'CB');
            dlYPred= modelLSTM(dlX,parameters);
            
            YPredValidation = single(gather(extractdata(dlYPred) > labelThreshold));
            %         clear res
            %         if min(test_target(:))==-1;
            %             YPredValidation(find(YPredValidation==0))=-1;
            %         end
            %         res(1:5) = Evaluation(YPredValidation,gather(extractdata(dlYPred)),test_target');
            
            save(strcat(URI,'LSTMadam_',num2str(id),'_',num2str(datas),'_',num2str(fold),'_',num2str(itero),'_',num2str(reitero),'.mat'),'dlYPred','parameters')
            
            
        end
    end
end





