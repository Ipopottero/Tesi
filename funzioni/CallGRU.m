
for itera=1:5
    
    %shuffle features
    if sh==1
        idx = randperm(size(train_data,2));
        train_data = train_data(:,idx);
        test_data = test_data(:,idx);
    end
    
    embeddingDimension = 1;
    
    numHiddenUnits = k;%min([500 size(train_target,2)*3]);
    
    inputSize = size(train_data,2) + 1;
    numClasses=size(train_target,2);
    
    parameters = struct;
    parameters.emb.Weights = dlarray(randn([embeddingDimension inputSize]));
    
    parameters.gru.InputWeights = dlarray(initializeGlorot(3*numHiddenUnits,embeddingDimension));
    parameters.gru.RecurrentWeights = dlarray(initializeGlorot(3*numHiddenUnits,numHiddenUnits));
    parameters.gru.Bias = dlarray(zeros(3*numHiddenUnits,1,'single'));
    
    parameters.fc.Weights = dlarray(initializeGaussian([numClasses,numHiddenUnits]));
    parameters.fc.Bias = dlarray(zeros(numClasses,1,'single'));
    
    miniBatchSize = 30;
    learnRate = 0.01;
    gradientDecayFactor = 0.5;
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
    
    % Loop over epochs.
    for epoch = 1:numEpochs
        
        % Loop over mini-batches.
        for i = 1:numIterationsPerEpoch
            iteration = iteration + 1;
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
            
            % Read mini-batch of data and convert the labels to dummy
            % variables.
            X = train_data(idx,:);
            
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
            
            [gradients,loss,dlYPred] = dlfeval(@modelGradients, dlX, labels, parameters);
            
            % Gradient clipping.
            gradients = dlupdate(@(g) thresholdL2Norm(g, gradientThreshold),gradients);
            
            if iteration>1
                gradOld=gradients;
            end
            if iteration==1
                gradOld=gradients;
            end
            
            % Update the network parameters using the Adam optimizer.
            epsilon = 1e-8;
            [parameters,trailingAvg,trailingAvgSq] = adamupdateStoc(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor, epsilon,id-1,gradOld);
            
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
    if itera==1
        dlYPredFIN= model(dlX,parameters);
    else
        dlYPredFIN= dlYPredFIN+model(dlX,parameters);
    end
    
end
