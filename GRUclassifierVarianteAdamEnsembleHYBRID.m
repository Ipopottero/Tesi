% function GRUclassifier(train_data,train_target, test_data,test_target)
%https://it.mathworks.com/help/deeplearning/ug/multilabel-text-classification-using-deep-learning.html
close all force

varianti = [2 6 11 26];

reitero = 3;
numEpochs = 150;

for itera = 1:40

    %     fid=fopen(strcat(URI,'GRU_variantEnsemble_',num2str(datas),'_',num2str(fold),'_',num2str(itera),'_',num2str(reitero),'.mat'));
    %     if fid>0
    %         continue;
    %     end

    %random variant
    for layer = 1:146
        [in choose] = min(rand(4, 1));
        Variant(layer) = varianti(choose);
    end

    % parametri GRU
    embeddingDimension = 1;
    numHiddenUnits = 50;
    inputSize = size(train_data, 2) + 1;
    numClasses = size(train_target, 2);

    parameters = struct;

    parameters.emb.Weights = dlarray(randn([embeddingDimension inputSize]));

    parameters.gru.InputWeights = dlarray(initializeGlorot(3 * numHiddenUnits, embeddingDimension));
    parameters.gru.RecurrentWeights = dlarray(initializeGlorot(3 * numHiddenUnits, numHiddenUnits));
    parameters.gru.Bias = dlarray(zeros(3 * numHiddenUnits, 1, 'single'));

    parameters.fc.Weights = dlarray(initializeGaussian([numClasses, numHiddenUnits]));
    parameters.fc.Bias = dlarray(zeros(numClasses, 1, 'single'));

    % parametri Temporal
    numBlocks = 4;
    numFilters = 175;
    filterSize = 3;
    dropoutFactor = 0.05;

    hyperparameters = struct;
    hyperparameters.NumBlocks = numBlocks;
    hyperparameters.DropoutFactor = dropoutFactor;

    %numChannels = numHiddenUnits;
    numChannels = size(train_target, 2);

    for k = 1:numBlocks
        parametersBlock = struct;
        blockName = "Block" + k;

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

        parameters.tcn.(blockName) = parametersBlock;
    end

    weights = initializeGaussian([numClasses, numChannels]);
    bias = zeros(numClasses, 1, 'single');

    parameters.tcn.fc.Weights = dlarray(weights);
    parameters.tcn.fc.Bias = dlarray(bias);

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
    numObservationsTrain = size(train_data, 1);
    numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
    validationFrequency = numIterationsPerEpoch;

    executionEnvironment = "auto";

    if plots == "training-progress"
        figure

        % Labeling F-Score.
        subplot(2, 1, 1)
        lineLossTrain = animatedline('Color', [0 0.447 0.741]);

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
            idx = (i - 1) * miniBatchSize + 1:i * miniBatchSize;

            % Read mini-batch of data and convert the labels to dummy
            % variables.
            X = train_data(idx, :);

            %labels (1,0), numClasses*miniBatchSize
            labels = train_target(idx, :)';

            if min(train_target(:)) == -1
                labels(find(labels == -1)) = 0;
            end

            % Convert mini-batch of data to dlarray.
            dlX = dlarray(X, 'BTC');

            % If training on a GPU, then convert data to gpuArray.
            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                dlX = gpuArray(dlX);
            end

            % Evaluate the model gradients, state, and loss using dlfeval and the
            % modelGradients function.

            [gradients, loss, dlYPred] = dlfeval(@modelGradientsHYBRID, dlX, labels, parameters, hyperparameters);

            % Gradient clipping.
            if iteration > 1
                gradOld = gradients;
            end

            gradients = dlupdate(@(g) thresholdL2Norm(g, gradientThreshold), gradients);

            if iteration == 1
                gradOld = gradients;
            end

            % Update the network parameters using the Adam optimizer.
            epsilon = 1e-8;
            id = 1;
            [parameters, trailingAvg, trailingAvgSq] = adamupdateStocEnsemble(parameters, gradients, ...
                trailingAvg, trailingAvgSq, iteration, learnRate, gradientDecayFactor, squaredGradientDecayFactor, epsilon, id - 1, gradOld, Variant);

            % Display the training progress.
            score = labelingFScore(extractdata(dlYPred) > labelThreshold, labels);

            addpoints(lineLossTrain, iteration, double(gather(score)))

            drawnow

        end

        % Shuffle data.
        idx = randperm(numObservationsTrain);
        train_data = train_data(idx, :);
        train_target = train_target(idx, :);
    end

    %test set
    dlX = dlarray(test_data, 'BTC');
    dlYPred = modelHYBRID(dlX, parameters, hyperparameters);

    %         YPredValidation = single(gather(extractdata(dlYPred) > labelThreshold));
    %         clear res
    %         if min(test_target(:))==-1;
    %             YPredValidation(find(YPredValidation==0))=-1;
    %         end
    %         res(1:5) = Evaluation(YPredValidation,gather(extractdata(dlYPred)),test_target');

    save(strcat(URI, 'GRU_variantEnsembleHYBRID_', num2str(datas), '_', num2str(fold), '_', num2str(itera), '_', num2str(reitero), '.mat'), 'dlYPred', 'parameters')

end
