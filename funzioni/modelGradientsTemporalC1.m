function [gradients,loss,dlY] = modelGradientsTemporalC1(dlX,T,parameters,hyperparameters)

dlY = modelTemporalC1(dlX,parameters,hyperparameters,true);

loss = crossentropy(dlY,T,'TargetCategories','independent');

gradients = dlgradient(mean(loss),parameters);

end