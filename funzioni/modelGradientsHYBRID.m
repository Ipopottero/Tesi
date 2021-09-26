function [gradients,loss,dlYPred] = modelGradientsHYBRID(dlX,T,parameters,hyperparameters)

dlYPred = modelHYBRID(dlX,parameters,hyperparameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end