function [XTransformed,YTransformed,numTimeSteps] = preprocessMiniBatch(XCell,YCell) 

    numTimeSteps = cellfun(@(sequence) size(sequence,2),XCell);
    sequenceLength = max(cellfun(@(sequence) size(sequence,2),XCell));
    
    miniBatchSize = numel(XCell);
    numFeatures = size(XCell{1},1);    
    classes = categories(YCell{1});
    numClasses = numel(classes);
    
    szX = [numFeatures miniBatchSize sequenceLength];
    XTransformed = zeros(szX,'single');
    
    szY = [numClasses miniBatchSize sequenceLength];
    YTransformed = zeros(szY,'single');
    
    for i = 1:miniBatchSize
        predictors = XCell{i};
        
        responses = onehotencode(YCell{i},1);
        
        % Left pad.
        XTransformed(:,i,:) = leftPad(predictors,sequenceLength);
        YTransformed(:,i,:) = leftPad(responses,sequenceLength);
        
    end

end