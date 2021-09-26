function [dlY,dlYprePool,dlYpostPool] = modelDCT(dlX,parameters)

% GRU
inputWeights = parameters.gru.InputWeights;
recurrentWeights = parameters.gru.RecurrentWeights;
bias = parameters.gru.Bias;

numHiddenUnits = size(inputWeights,1)/3;
hiddenState = dlarray(zeros([numHiddenUnits 1]));
 
dlY = gru(dlX, hiddenState, inputWeights, recurrentWeights, bias);
   
dlY = avgpool(dlY,[],3);
% 
% for i=1:numHiddenUnits
%     for j=1:size(dlYb,2)
%         A=dct2(dlYb(i,j,:));
%         dlY(i,j,1)=dlarray(max(A(2:end)));
%     end
% end


% Fully connect
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
dlY = fullyconnect(dlY,weights,bias);

% Sigmoid
dlY = sigmoid(dlY);

end