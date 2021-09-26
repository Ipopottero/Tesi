function [dlY,dlYprePool,dlYpostPool] = model_C1AWGDEEP(dlX,parameters)

%conv1
weights = parameters.conv1.Weights;
bias = parameters.conv1.Bias;
dlX = dlconv(dlX,weights,bias,'Padding','same');

%avgpool
poolSize = 3;
dlX = avgpool(dlX,poolSize,'Padding','same','PoolFormat','T');


% GRU
inputWeights = parameters.gru.InputWeights;
recurrentWeights = parameters.gru.RecurrentWeights;
bias = parameters.gru.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));

dlY = gru(dlX, hiddenState, inputWeights, recurrentWeights, bias);

%second gru
inputWeights = parameters.gru2.InputWeights;
recurrentWeights = parameters.gru2.RecurrentWeights;
bias = parameters.gru2.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));
dlY = gru(dlY, hiddenState, inputWeights, recurrentWeights, bias);

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