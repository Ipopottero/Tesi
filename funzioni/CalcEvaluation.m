function [res,metodo]=CalcEvaluation(score,test_target,numClasses,metodo,labelThreshold)

if nargin==4
    labelThreshold=1/numClasses*2;
end

YPredValidation = single(score > (labelThreshold));
if min(test_target(:))==-1;
    YPredValidation(find(YPredValidation==0))=-1;
end

if size(YPredValidation,1)==size(test_target,1)
    res = Evaluation(YPredValidation',score',test_target');
else
    res = Evaluation(YPredValidation,score,test_target');
end
% evluation for MLC algorithms, there are fifteen evaluation metrics
% 
% syntax
%   ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   Pre_Labels          - L x num_test data matrix of predicted labels
%   Outputs             - L x num_test data matrix of scores
metodo=metodo+1;
