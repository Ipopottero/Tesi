function [p, avg_g, avg_gsq] = adamupdateStocEnsemble(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant)
%ADAMUPDATE Update parameters via adaptive moment estimation
%
%   [NET,AVG_G,AVG_SQG] = ADAMUPDATE(NET,GRAD,AVG_G,AVG_SQG,ITER) updates
%   the learnable parameters of the dlnetwork NET using the Adam gradient
%   descent algorithm, with a default global learning rate of 0.001, a
%   gradient decay factor of 0.9 and a squared gradient decay factor of
%   0.999. Use ADAMUPDATE to iteratively update learnable parameters during
%   network training.
%
%   Input GRAD contains the gradients of the loss with respect to each of
%   the network parameters. Inputs AVG_G and AVG_SQG contain, the moving
%   average of the parameter gradients and the moving average of the
%   element-wise squares of the parameter gradients, respectively. AVG_G
%   and AVG_SQG are obtained from the previous call to ADAMUPDATE. GRAD,
%   AVG_G, and AVG_SQG must be tables with the same structure as
%   NET.Learnables, with a Value variable containing a cell array of
%   parameter gradients, average gradients, or average squared gradients.
%   Input GRAD can be obtained using the dlgradient and dlfeval functions.
%   The global learning rate is multiplied by the corresponding learning
%   rate factor for each parameter in each layer of the dlnetwork.
%
%   If inputs AVG_G and AVG_SQG are empty, the function assumes no previous
%   gradients and executes like the first update in a series of iterations.  
%
%   Input ITER contains the update iteration number. ITER must be a
%   positive integer. Use a value of 1 for the first call to ADAMUPDATE and
%   increment by 1 for each successive call in a series of iterations. The
%   Adam algorithm uses this value to correct for bias in the moving
%   averages at the beginning of a set of iterations.
%
%   Outputs NET, AVG_G, and AVG_SQG are the updated dlnetwork, average
%   gradients, and average squared gradients, respectively.
%
%   [PARAMS,AVG_G,AVG_SQG] = ADAMUPDATE(PARAMS,GRAD,AVG_G,AVG_SQG,ITER)
%   updates the deep learning parameters in PARAMS using the Adam gradient
%   descent algorithm, with a default learning rate of 0.001, a gradient
%   decay factor of 0.9, and a squared gradient decay factor of 0.999.
%   Input PARAMS can be a dlarray, a numeric array, a cell array, a
%   structure, or a table with a Value variable containing the learnable
%   parameters of the network. GRAD, AVG_G, and AVG_SQG must have the same
%   datatype and ordering as PARAMS. Input GRAD can be obtained using the
%   dlgradient and dlfeval functions.  All parameter values are updated
%   using the global learning rate.
%
%   Outputs PARAMS, AVG_G, and AVG_SQG are the updated parameters, average
%   gradients, and average squared gradients, respectively.
%
%   [___] = ADAMUPDATE(___,LEARNRATE,G_DECAY,SQG_DECAY) also specifies
%   values to use for the global learning rate, gradient decay factor, and
%   squared gradient decay factor. LEARNRATE must be a positive scalar.
%   G_DECAY and SQG_DECAY must be scalars between 0 and 1.
%
%   [___] = ADAMUPDATE(___,LEARNRATE,G_DECAY,SQG_DECAY,EPSILON) specifies a
%   small constant used to prevent division by zero in the update equation.
%   The default value of EPSILON is 1e-8.
%
%   Example 1:
%      % Perform an Adam update step with learning rate of 0.01, gradient 
%      % decay factor of 0.9, and squared gradient decay factor of 0.95.
%      p = rand(3,3,4);
%      avg_g = ones(3,3,4);
%      avg_sqg = ones(3,3,4); 
%      g = ones(3,3,4);
%      iteration = 15;
%      [p,avg_g,avg_sqg] = adamupdate(p,g,avg_g,avg_sqg,iteration,0.01,0.9,0.95);
%
%   Example 2:
%      % Perform a single epoch of training on a network that classifies 
%      % handwritten digits.
%
%      % Load training data and encode.
%      [XTrain,YTrain] = digitTrain4DArrayData;
%      XTrain = single(XTrain);
%      YEncoded = onehotencode(YTrain,2)';
%      
%      % Define the network and specify the average image using the 'Mean' option 
%      % in the image input layer.
%      layers = [
%          imageInputLayer([28 28 1],'Name','input','Mean',mean(XTrain,4))
%          convolution2dLayer(5,20,'Name','conv1')
%          reluLayer('Name','relu1')
%          convolution2dLayer(3,20,'Padding',1,'Name','conv2')
%          reluLayer('Name','relu2')
%          fullyConnectedLayer(size(YEncoded, 1),'Name','fc')];
%      net = dlnetwork(layerGraph(layers));
%      
%      % Initialize average parameter gradients and iteration.
%      gradientsAvg = [];
%      squaredGradientsAvg = [];
%      iter = 0;
%
%      % Loop over one epoch of mini-batches.
%      miniBatchSize = 128;
%      for i = 1:miniBatchSize:numel(YTrain)
%          % Read mini-batch of data and convert the labels to dummy variables.
%          idx = i:min(i+miniBatchSize-1,numel(YTrain));
%          
%          % Convert mini-batch of data to dlarray.
%          X = dlarray(XTrain(:,:,:,idx),'SSCB');
%          
%          % Evaluate the model gradients using dlfeval and the
%          % modelGradients function defined below.
%          gradients = dlfeval(@modelGradients,net,X,YEncoded(:,idx));
%          
%          % Update the network parameters using the Adam algorithm.
%          iter = iter + 1;
%          [net.Learnables,gradientsAvg,squaredGradientsAvg] = adamupdate(...
%              net.Learnables,gradients,gradientsAvg,squaredGradientsAvg,iter);
%      end
%
%      function gradients = modelGradients(net,X,Y)
%          loss = crossentropy(softmax(forward(net,X)),Y);
%          gradients = dlgradient(loss,net.Learnables);
%      end
%
%   See also DLFEVAL, DLGRADIENT, DLNETWORK, DLUPDATE, RMSPROPUPDATE, SGDMUPDATE

%   Copyright 2019-2020 The MathWorks, Inc.
    
global contatore
contatore=1;
    
global contatore2
contatore2=1;

if nargin<9
    epsilon = 1e-8;
else
    validateattributes(epsilon, "numeric", {'scalar','real','finite','positive'}, ...
        'adamupdate', 'EPSILON', 9);
end

if nargin<8
    beta2 = 0.999;
else
    validateattributes(beta2, "numeric", {'scalar','real','finite','>=',0,'<',1}, ...
        'adamupdate', 'GSQ_DECAY', 8);
end

if nargin<7
    beta1 = 0.9;
else
    validateattributes(beta1, "numeric", {'scalar','real','finite','>=',0,'<',1}, ...
        'adamupdate', 'G_DECAY', 7);
end

if nargin<6
    lr = 0.001;
else
    validateattributes(lr, "numeric", {'scalar', 'real', 'finite', 'nonnegative'}, ...
        'adamupdate', 'LR', 6);
end

% Validate that iteration is a positive integer
validateattributes(t, "numeric", {'scalar', 'real', 'finite', 'positive', 'integer'}, ...
        'adamupdate', 'T', 5);

if isempty(avg_g) && isempty(avg_gsq)
    
    % Execute a first-step update with g_av and g_sq_av set to 0.  The step
    % will create arrays for these that are the correct size
    func = deep.internal.LearnableUpdateFunction( ...
        @(p, g) iSingleStepValue(p, g, 0, 0, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant), ...
        @(p, g) iSingleStepParameter(p, g, 0, 0, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant) );
    
    [p, avg_g, avg_gsq] = deep.internal.networkContainerFun(func, p, g);
else
    % Execute the normal update
    func = deep.internal.LearnableUpdateFunction( ...
        @(p, g, avg_g, avg_gsq) iSingleStepValue(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant), ...
        @(p, g, avg_g, avg_gsq) iSingleStepParameter(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant) );
    
    [p, avg_g, avg_gsq] = deep.internal.networkContainerFun(func, p, g, avg_g, avg_gsq);
end
end


function [p, avg_g, avg_gsq] = iSingleStepParameter(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant)
% Apply per-parameter learn-rate factor
lr = lr .* p.LearnRateFactor;

global contatore
contatore=contatore+1;
    
Y=WeightLR_ml(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,Variant(contatore-1)-1,gradOld);

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
biasCorrection = sqrt(1-beta2.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr.*Y;

[step, avg_g, avg_gsq] = nnet.internal.cnn.solver.adamstep(...
                        g, avg_g, avg_gsq, effectiveLearnRate, beta1, beta2, epsilon);
p.Value = p.Value + step;
end

function [p, avg_g, avg_gsq] = iSingleStepValue(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,approccio,gradOld,Variant)

global contatore2
contatore2=contatore2+1;

Y=WeightLR_ml(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,Variant(contatore2-1)-1,gradOld);

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
biasCorrection = sqrt(1-beta2.^t)./(1-beta1.^t);
effectiveLearnRate = biasCorrection.*lr.*Y;

[step, avg_g, avg_gsq] = nnet.internal.cnn.solver.adamstep(...
                        g, avg_g, avg_gsq, effectiveLearnRate, beta1, beta2, epsilon);
p = p + step;
end
