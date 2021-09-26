function [gradients,loss,dlY] = modelGradientsTemporal(dlX,T,parameters,hyperparameters)

dlY = modelTemporal(dlX,parameters,hyperparameters,true);

loss = crossentropy(dlY,T,'TargetCategories','independent');

gradients = dlgradient(mean(loss),parameters);

end