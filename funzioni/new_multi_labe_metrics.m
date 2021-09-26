function [Absolute_false,Coverage,Absolute_true,Aiming,Accuracy]=new_multi_labe_metrics(Pre_Labels,test_target)
% output five multi-label Metrics
% multi_labe_metrics takes,
%       Pre_Labels   -  Multi-label predicted by ML-GKR, A QxM2 array, if the ith testing 
%                          instance belongs to the jth class, test_target(j,i) equals +1, 
%                           otherwise test_target(j,i) equals -1
%       test_target   -    A QxM2 array, if the ith testing instance belongs to the jth class, 
%                          test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%     
% and returns,
%       Absolute_false - refer to [1] for detailed description
%       Coverage - refer to [1] for detailed description
%       Absolute_true - refer to [1] for detailed description
%       Aiming - refer to [1] for detailed description
%       Accuracy - refer to [1] for detailed description
%[1] Cheng X, Zhao SG, Xiao X, et al. iATC-mISF: a multi-label classifier 
%     for predicting the classes of anatomical therapeutic chemicals.[J]. 
%     Bioinformatics (Oxford, England), 2016.

%compare Pre_Labels and test_target,ACC
res=Pre_Labels-test_target;
 res=abs(res);
res=res./2;
[leiNum,instance_num]=size(Pre_Labels);
% Absolute true
acc=0;
for i=1:instance_num
    if(0==sum(res(:,i)))
        acc=acc+1;
    end
end
Absolute_true=acc/instance_num;

Lab=test_target+ones(leiNum,instance_num);
Lab=Lab./2;
preLab=Pre_Labels+ones(leiNum,instance_num);
preLab=preLab./2;

% Accuracy
fenzi=preLab&Lab;
fenmu=preLab|Lab;

mlACC=0;
for i=1:instance_num
    somma=sum(fenzi(:,i))/sum(fenmu(:,i));
    somma(isnan(somma))=0;
    mlACC=mlACC+somma;
end
Accuracy=mlACC/instance_num;

% Aiming
fenzi=preLab&Lab;
fenmu=preLab;
mlAIM=0;

for i=1:instance_num
    mlDen=sum(fenmu(:,i));
    if mlDen==0 
        mlDen=1;
    end   
    mlAIM=mlAIM+(sum(fenzi(:,i))/mlDen);
end

Aiming=mlAIM/instance_num;


% Coverage
fenzi=preLab&Lab;
fenmu=Lab;
mlCov=0;
for i=1:instance_num
    somma=sum(fenzi(:,i))/sum(fenmu(:,i));
    somma(isnan(somma))=0;
    mlCov=mlCov+somma;
end
Coverage=mlCov/instance_num;

%absolute false
fenzi=preLab&Lab;
fenmu=preLab|Lab;
abFalse=0;
for i=1:instance_num
    fen=(sum(fenmu(:,i))-sum(fenzi(:,i)))/leiNum;
    abFalse=abFalse+fen;
end
Absolute_false=abFalse/instance_num;


