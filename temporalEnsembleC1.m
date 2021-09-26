close all force
reitero=2;
numEpochs = 100;
varianti=[2 6 11 26];

for itera=1:40
    
    
    %random variant
    for layer=1:160
        [in choose]=min(rand(4,1));
        Variant(layer)=varianti(choose);
    end
    
    numBlocks = 4;
    numFilters = 175;
    filterSize = 3;
    dropoutFactor = 0.05;
    numClasses = size(train_target,2);
    
    hyperparameters = struct;
    hyperparameters.NumBlocks = numBlocks;
    hyperparameters.DropoutFactor = dropoutFactor;
    
    
    parameters = struct;
    numChannels = 1;
    
    % conv1
    filterSize = 3;
    nFilter = 1;
    arr = [nFilter filterSize numChannels];
    nOut = prod(filterSize)*nFilter;
    nIn = prod(filterSize)*nFilter;
    aaa = initializeGlorotATT(arr,nOut,nIn);
    parameters.conv1.Weights = dlarray(aaa,'UTC');
    parameters.conv1.Bias = dlarray(zeros(nFilter,1,'single'));
    
    % residual blocks
    for k = 1:numBlocks
        parametersBlock = struct;
        blockName = "Block"+k;
        
        weights = initializeGaussian([filterSize, numChannels, numFilters]);
        bias = zeros(numFilters, 1, 'single');
        parametersBlock.Conv1.Weights = dlarray(weights);
        parametersBlock.Conv1.Bias = dlarray(bias);
        
        weights = initializeGaussian([filterSize, numFilters, numFilters]);
        bias = zeros(numFilters, 1, 'single');
        parametersBlock.Conv2.Weights = dlarray(weights);
        parametersBlock.Conv2.Bias = dlarray(bias);
        
        % If the input and output of the block have different numbers of
        % channels, then add a convolution with filter size 1.
        if numChannels ~= numFilters
            weights = initializeGaussian([1, numChannels, numFilters]);
            bias = zeros(numFilters, 1, 'single');
            parametersBlock.Conv3.Weights = dlarray(weights);
            parametersBlock.Conv3.Bias = dlarray(bias);
        end
        numChannels = numFilters;
        
        parameters.(blockName) = parametersBlock;
    end
    
    weights = initializeGaussian([numClasses,numChannels]);
    bias = zeros(numClasses,1,'single');
    
    parameters.FC.Weights = dlarray(weights);
    parameters.FC.Bias = dlarray(bias);
    
    maxEpochs = 30;
    miniBatchSize = 30;
    learnRate = 0.01;
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.999;
    initialLearnRate = 0.001;
    learnRateDropFactor = 0.1;
    learnRateDropPeriod = 12;
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
            
            [gradients,loss,dlYPred] = dlfeval(@modelGradientsTemporalC1,dlX,dlarray(labels),parameters,hyperparameters);
            
            % Clip the gradients.
            gradients = dlupdate(@(g) thresholdL2Norm(g,gradientThreshold),gradients);
            
            if iteration>1
                gradOld=gradients;
            end
            if iteration==1
                gradOld=gradients;
            end
            
            % Update the network parameters using the Adam optimizer.
            id=1;
            epsilon = 1e-8;
            [parameters,trailingAvg,trailingAvgSq] = adamupdateStocEnsemble(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor, epsilon,id-1,gradOld,Variant);
            
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
    dlYPred= modelTemporalC1(dlX,parameters,hyperparameters,1);
    
    YPredValidation = single(gather(extractdata(dlYPred) > labelThreshold));
    res(1:10) = Evaluation(YPredValidation,gather(extractdata(dlYPred)),test_target');
    
    save(strcat(URI,'TEMPORAL_ensembleC1_',num2str(id),'_',num2str(datas),'_',num2str(fold),'_',num2str(itera),'_',num2str(reitero),'.mat'),'dlYPred')
    
end



