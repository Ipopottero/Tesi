function [gradients,loss,dlYPred] = modelGradients_C1(dlX,T,parameters)

dlYPred = model_C1(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end