%%
numChannels = 5;
miniBatchSize = 128;
sequenceLength = 512;
X = rand(numChannels,miniBatchSize,sequenceLength);
dlX = dlarray(X,'CBT');
%%
filterSize = 3;
numFilters = 64;
weights = rand(filterSize,numChannels,numFilters);
bias = zeros(1,numFilters);
%%
dlY = dlconv(dlX,weights,bias,'WeightsFormat','TCU');