function [gradients,loss,dlYPred] = modelGradientsAttention(dlX,T,parameters)

dlYPred = modelAttention(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end