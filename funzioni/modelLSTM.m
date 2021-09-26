function [dlY,dlYprePool,dlYpostPool] = modelLSTM(dlX,parameters)


% LSTM
% Create formatted dlarrays for the lstm parameters with three
% hidden units.
numObservations=size(dlX,2);
numFeatures=size(dlX,1);
% H0 = dlarray(randn(numHiddenUnits,numObservations),'CB');
% C0 = dlarray(randn(numHiddenUnits,numObservations),'CB');
% weights = dlarray(randn(4*numHiddenUnits,numFeatures),'CU');
% recurrent = dlarray(randn(4*numHiddenUnits,numFeatures),'CU');
% bias = dlarray(randn(4*numHiddenUnits,1),'C');
weights = parameters.lstm.weights;
recurrent = parameters.lstm.recurrent;
bias = parameters.lstm.bias;
H0 = parameters.lstm.H0;
C0 = parameters.lstm.C0;

%       % Apply an lstm calculation
dlY = lstm(dlX,H0,C0,weights,recurrent,bias);
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