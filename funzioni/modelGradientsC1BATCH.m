function [gradients,loss,dlYPred] = modelGradientsC1BATCH(dlX,T,parameters)

dlYPred = modelC1BATCH(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end