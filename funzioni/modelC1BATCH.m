function [dlY,dlYprePool,dlYpostPool] = modelC1BATCH(dlX,parameters)

% conv1
weights = parameters.conv1.Weights;
bias = parameters.conv1.Bias;
dlX = dlconv(dlX,weights,bias,'Padding','same');

% Normalization layer
dim = find(dims(dlX)=='T');
mu = mean(dlX,dim);
sigmaSq = var(dlX,1,dim);
epsilon = 1e-5;
dlX = (dlX - mu) ./ sqrt(sigmaSq + epsilon);

% GRU
inputWeights = parameters.gru.InputWeights;
recurrentWeights = parameters.gru.RecurrentWeights;
bias = parameters.gru.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));

dlY = gru(dlX, hiddenState, inputWeights, recurrentWeights, bias);
dlYprePool=dlY;

% Max pooling along time dimension
dlY = max(dlY,[],3);
dlYpostPool=dlY;

% Fully connect
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
dlY = fullyconnect(dlY,weights,bias);

% Sigmoid
dlY = sigmoid(dlY);

end