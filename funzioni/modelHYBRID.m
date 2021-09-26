function [dlY,dlYprePool,dlYpostPool] = modelHYBRID(dlX,parameters, hyperparameters)

% GRU ------------------------------------------------------------
inputWeights = parameters.gru.InputWeights;
recurrentWeights = parameters.gru.RecurrentWeights;
bias = parameters.gru.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));

dlY = gru(dlX, hiddenState, inputWeights, recurrentWeights, bias);

% Fully connect
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
dlY = fullyconnect(dlY,weights,bias);

% Max pooling along time dimension
% dlY = max(dlY,[],3);

% Sigmoid
dlY = sigmoid(dlY);

% TCN ----------------------------------------------------------------
numBlocks = hyperparameters.NumBlocks;
dropoutFactor = hyperparameters.DropoutFactor;

% Residual blocks.
for k = 1:numBlocks
    dilationFactor = 2^(k-1);
    parametersBlock = parameters.tcn.("Block"+k);
    
    dlY = residualBlock(dlY,dilationFactor,dropoutFactor,parametersBlock,'true');
end

% Fully connect
weights = parameters.tcn.fc.Weights;
bias = parameters.tcn.fc.Bias;
dlY = fullyconnect(dlY,weights,bias);

dlYprePool = dlY;

% Max pooling along time dimension
dlY = max(dlY,[],3);

dlYpostPool = dlY;

% Sigmoid
dlY = sigmoid(dlY);

end