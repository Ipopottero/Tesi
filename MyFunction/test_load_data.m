clear all
clc

datas = 2;
[NF, test_data, test_target, train_data, train_target] = load_data(datas);

assert (isequal(size(test_data),[100,68]))
%%

% Fin qua tutto okay