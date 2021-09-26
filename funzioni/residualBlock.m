function dlY = residualBlock(dlX,dilationFactor,dropoutFactor,parametersBlock,doTraining)

% Convolution options.
filterSize = size(parametersBlock.Conv1.Weights,1);
paddingSize = (filterSize - 1) * dilationFactor;

% Convolution.
weights = parametersBlock.Conv1.Weights;
bias =  parametersBlock.Conv1.Bias;
dlY = dlconv(dlX,weights,bias, ...
    'WeightsFormat','TCU', ...
    'DilationFactor', dilationFactor, ...
    'Padding', [paddingSize; 0]);

% Normalization.
dim = find(dims(dlY)=='T');
mu = mean(dlY,dim);
sigmaSq = var(dlY,1,dim);
epsilon = 1e-5;
dlY = (dlY - mu) ./ sqrt(sigmaSq + epsilon);

% ReLU, spatial dropout.
dlY = relu(dlY);
dlY = spatialDropout(dlY,dropoutFactor,doTraining);

% Convolution.
weights = parametersBlock.Conv2.Weights;
bias = parametersBlock.Conv2.Bias;
dlY = dlconv(dlY,weights,bias, ...
    'WeightsFormat','TCU', ...
    'DilationFactor', dilationFactor, ...
    'Padding',[paddingSize; 0]);

% Normalization.
dim = find(dims(dlY)=='T');
mu = mean(dlY,dim);
sigmaSq = var(dlY,1,dim);
epsilon = 1e-5;
dlY = (dlY - mu) ./ sqrt(sigmaSq + epsilon);

% ReLU, spatial dropout.
dlY = relu(dlY);
dlY = spatialDropout(dlY,dropoutFactor,doTraining);

% Optional 1-by-1 convolution.
if ~isequal(size(dlX),size(dlY))
    weights = parametersBlock.Conv3.Weights;
    bias = parametersBlock.Conv3.Bias;
    dlX = dlconv(dlX,weights,bias,'WeightsFormat','TCU');
end

% Addition and ReLU
dlY = relu(dlX + dlY);

end