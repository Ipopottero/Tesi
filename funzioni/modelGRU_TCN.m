function [dlY,dlYprePool,dlYpostPool] = modelGRU_TCN(dlX,parameters, hyperparameters)

% GRU ------------------------------------------------------------
inputWeights = parameters.parametersGRU.gru.InputWeights;
recurrentWeights = parameters.parametersGRU.gru.RecurrentWeights;
bias = parameters.parametersGRU.gru.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));

dlY = gru(dlX, hiddenState, inputWeights, recurrentWeights, bias);
% dlYprePool=dlY;

% Max pooling along time dimension
dlY = max(dlY,[],3);
% dlYpostPool=dlY;

% Fully connect
weights = parameters.parametersGRU.fc.Weights;
bias = parameters.parametersGRU.fc.Bias;
dlY = fullyconnect(dlY,weights,bias);

% Sigmoid
dlY = sigmoid(dlY);

% TCN ----------------------------------------------------------------
numBlocks = hyperparameters.NumBlocks;
dropoutFactor = hyperparameters.DropoutFactor;

%dlY = dlX;

% Residual blocks.
for k = 1:numBlocks
    dilationFactor = 2^(k-1);
    parametersBlock = parameters.parametersTCN.("Block"+k);
    
    dlY = residualBlock(dlY,dilationFactor,dropoutFactor,parametersBlock,1);
end

% Fully connect
weights = parameters.parametersTCN.FC.Weights;
bias = parameters.parametersTCN.FC.Bias;
dlY = fullyconnect(dlY,weights,bias);

dlYprePool=dlY;
% Max pooling along time dimension
dlY = max(dlY,[],3);
dlYpostPool=dlY;
% Sigmoid
dlY = sigmoid(dlY);

end