function dlY = modelTemporalC1(dlX,parameters,hyperparameters,doTraining)

numBlocks = hyperparameters.NumBlocks;
dropoutFactor = hyperparameters.DropoutFactor;

%conv1
weights = parameters.conv1.Weights;
bias = parameters.conv1.Bias;
dlX = dlconv(dlX,weights,bias,'Padding','same');

dlY = dlX;

% Residual blocks.
for k = 1:numBlocks
    dilationFactor = 2^(k-1);
    parametersBlock = parameters.("Block"+k);
    
    dlY = residualBlockC1(dlY,dilationFactor,dropoutFactor,parametersBlock,doTraining);
end

% Fully connect
weights = parameters.FC.Weights;
bias = parameters.FC.Bias;
dlY = fullyconnect(dlY,weights,bias);

% Max pooling along time dimension
dlY = max(dlY,[],3);

% Sigmoid
dlY = sigmoid(dlY);

end