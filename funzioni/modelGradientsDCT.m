function [gradients,loss,dlYPred] = modelGradientsDCT(dlX,T,parameters)

dlYPred = modelDCT(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end 