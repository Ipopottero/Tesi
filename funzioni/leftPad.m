function sequencePadded = leftPad(sequence,sequenceLength)

[numFeatures,numTimeSteps] = size(sequence);

paddingSize = sequenceLength - numTimeSteps;
padding = zeros(numFeatures,paddingSize);

sequencePadded = [padding sequence];

end