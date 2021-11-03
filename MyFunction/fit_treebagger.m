function [modelli] = fit_treebagger(train_data,train_target, n_trees)
% Utilizzo la classe TreeBagger per un unico label

n_labels = size(train_target, 2); 
modelli = {};

for i = [1:n_labels]
    mdl = TreeBagger(n_trees,train_data,train_target(:,i),'OOBPrediction','On',...
        'Method','classification');
    modelli{i} = mdl;
end

end




