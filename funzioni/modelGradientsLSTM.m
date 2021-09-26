function [gradients,loss,dlYPred] = modelGradientsLSTM(dlX,T,parameters)

dlYPred = modelLSTM(dlX,parameters);

loss = crossentropy(dlYPred,T,'TargetCategories','independent');

gradients = dlgradient(loss,parameters);

end