clc


[NF, test_data, test_target, train_data, train_target] = load_data(3);

modello  = fit_treebagger(train_data, train_target, 60);

save modello modello

%view(mdl.Trees{1},'Mode','graph')
%%

figure;
hold on
for i = [1:size(modello,2)]
     plot(oobError(modello{i}));
end

xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
