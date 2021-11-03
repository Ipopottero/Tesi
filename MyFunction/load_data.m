function [NF, test_data, test_target, train_data, train_target] = load_data(datas)
%load_data carica i dati

if datas==1
    load('cal500.mat')
    NF=1;
elseif datas==2
    load('image.mat')
    NF=1;
elseif datas==3
    load('scene.mat')
    NF=1;
elseif datas==4
    load('yeast.mat')
    NF=1;
elseif datas==5
    load('arts.mat')
    NF=1;
end

assert(size(train_data,2) == size(test_data,2))     %colonne uguali
assert(size(train_data,1) == size(train_target,1))  %righe ugualis