close all
clear all
clc


%[NF, test_data, test_target, train_data, train_target] = load_data(datas);
train_data = [ 1 2 3
    1 4 5];
    
train_target = [0
    1];
test_data=[ 1 2 3
    1 4 5];
test_target = [1
    0];

[test_data, test_target, train_data, train_target] = preprocessing(test_data, ...
    test_target, train_data, train_target, false);


assert(isequal(train_data,[2 3; 4 5]))
assert(isequal(test_data,[2 3; 4 5]))
assert(isequal(train_target,[-1;1]))
assert(isequal(test_target,[1;-1]))

display("funziona???")










