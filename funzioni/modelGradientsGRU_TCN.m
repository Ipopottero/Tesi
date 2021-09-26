function [gradients,loss,dlYPred] = modelGradientsGRU_TCN(dlX,T,parameters, hyperparameters)

dlYPred = modelGRU_TCN(dlX,parameters, hyperparameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end