function [gradients,loss,dlYPred] = modelGradients_C1AWGDEEP(dlX,T,parameters)

dlYPred = model_C1AWGDEEP(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end