function [dlY,dlYprePool,dlYpostPool] = modelAttention(dlX,parameters)


% GRU
inputWeights = parameters.gru.InputWeights;
recurrentWeights = parameters.gru.RecurrentWeights;
bias = parameters.gru.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));

[dlY hiddenState]= gru(dlX, hiddenState, inputWeights, recurrentWeights, bias);
dlYprePool=dlY;

% Max pooling along time dimension
dlY = max(dlY,[],3);
dlYpostPool=dlY;

% Attention.
keyboard
weights = parameters.decoder.attn.Weights;
[attentionScores, context] = attention(hiddenState, dlY, weights);

% Concatenate.
dlY = cat(1, dlY, repmat(context, [1 1 sequenceLength]));

% Fully connect
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
dlY = fullyconnect(dlY,weights,bias);

% Sigmoid
dlY = sigmoid(dlY);

end