function [gradients,loss,dlYPred] = modelGradients_C1AWG(dlX,T,parameters)

dlYPred = model_C1AWG(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end