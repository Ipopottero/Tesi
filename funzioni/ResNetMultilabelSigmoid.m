
gradDecay = 0.75;
sqGradDecay = 0.95;

for reitero=1:3
    if reitero==1
        numEpochs = 20;
    elseif reitero==2
        numEpochs = 50;
    elseif reitero==3
        numEpochs = 75;
    end
    
    numClasses=size(test_target,2);
    net = resnet50;
    lgraph = layerGraph(net);
    %lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');
    lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
        sigmoidLayer('Name','sigmoid')
        ];
    lgraph = addLayers(lgraph,newLayers);
    lgraph = connectLayers(lgraph,'avg_pool','fc');
    inputLayer = net.Layers(1);
    siz = inputLayer.InputSize(1:2);
    
    TR=  train_data;
    TE=  test_data;
    y=  train_target;
    yy=  test_target;
    
    
    %reshape vector -> matrix
    for itera=1:10
        
        for pattern=1:size(TR,1)
            network_features=TR(pattern,:);
            if pattern==1 && fold==1
                ord=randperm(length(network_features));
            end
            if mod(itera,2)==1 && itera<=15
                % Creare immagine riarrangiando in maniera random
                NW=ceil(length(network_features)^0.5);
                NR(1:NW^2)=0;
                NR(1:length(network_features))=network_features(ord);
                IM=reshape(NR,NW,NW);
                if size(IM,3)==1
                    IM(:,:,2)=IM;
                    IM(:,:,3)=IM(:,:,1);
                end
                %reshape multichannel
            elseif mod(itera,2)==0 && itera<=15
                NW=ceil(length(network_features)^0.5);
                NR(1:NW^2)=0;
                NR(1:length(network_features))=network_features(ord);
                IM = ReshapeLayerMultichannel(NR, NW, NW);
            elseif itera>15
                %% method named SUM in the paper
                I=network_features;
                IM=zeros(length(I),length(I));
                for ii=1:length(I)%I is a given image
                    for jj=1:length(I)
                        %random subset of 2-6 features
                        n=DimDCT(ii,jj);
                        %to reduce the subset to a single dimension
                        D=sum(I(OrdDct(ii,jj,1:n)));
                        IM(ii,jj)=D(1);
                    end
                end
                IM(:,:,2)=IM;
                IM(:,:,3)=IM(:,:,1);
                IM=imresize(IM,[250 250]);
                
            end
            Training{pattern}=IM;
        end
        
        for pattern=1:size(TE,1)
            network_features=TE(pattern,:);
            if mod(itera,2)==1 && itera<=15
                % Creare immagine riarrangiando in maniera random
                NW=ceil(length(network_features)^0.5);
                NR(1:NW^2)=0;
                NR(1:length(network_features))=network_features(ord);
                IM=reshape(NR,NW,NW);
                if size(IM,3)==1
                    IM(:,:,2)=IM;
                    IM(:,:,3)=IM(:,:,1);
                end
                %reshape multichannel
            elseif mod(itera,2)==0 && itera<=15
                NW=ceil(length(network_features)^0.5);
                NR(1:NW^2)=0;
                NR(1:length(network_features))=network_features(ord);
                IM = ReshapeLayerMultichannel(NR, NW, NW);
            elseif itera>15
                %% method named SUM in the paper
                I=network_features;
                IM=zeros(length(I),length(I));
                for ii=1:length(I)%I is a given image
                    for jj=1:length(I)
                        %random subset of 2-6 features
                        n=DimDCT(ii,jj);
                        %to reduce the subset to a single dimension
                        D=sum(I(OrdDct(ii,jj,1:n)));
                        IM(ii,jj)=D(1);
                    end
                end
                IM(:,:,2)=IM;
                IM(:,:,3)=IM(:,:,1);
                IM=imresize(IM,[250 250]);
                
            end
            Test{pattern}=IM;
        end
        
        YTrain=categorical(y);
        
        %creo il training set
        clear nome trainingImages
        trainingImages = zeros(siz(1),siz(2),3,length(length(Training)),'uint8');
        tmp=1;
        for pattern=1:length(Training)
            IM=Training{pattern};%singola data immagine
            
            IM=imresize(IM,[siz(1) siz(2)]);%si deve fare size immagini per rendere compatibili con CNN
            if size(IM,3)==1
                IM(:,:,2)=IM;
                IM(:,:,3)=IM(:,:,1);
            end
            
            trainingImages(:,:,:,tmp)=IM; label(tmp)=y(pattern);tmp=tmp+1;
        end
        imageSize=size(IM);
        
        YTrain=train_target;
        XTrain = single(trainingImages);
        
        %     for id=1%:7
        
        close all force
        
        % Initialize average parameter gradients and iteration.
        gradientsAvg = [];
        squaredGradientsAvg = [];
        averageGrad = [];
        averageSqGrad = [];
        iter = 0;
        plots=1;
        start=tic;
        
        iteration = 0;
        miniBatchSize = 30;
        numObservations = size(train_target,1);
        numIterationsPerEpoch = floor(numObservations./miniBatchSize);
        clear dlnet
        dlnet = dlnetwork(lgraph)
        
        
        figure
        lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
        ylim([0 inf])
        xlabel("Iteration")
        ylabel("Loss")
        grid on
        
        for epoch = 1:numEpochs
            % Shuffle data.
            idx = randperm(size(XTrain,4));
            XTrain = XTrain(:,:,:,idx);
            YTrain = YTrain(idx,:);
            
            for i = 1:numIterationsPerEpoch
                iteration = iteration + 1;
                
                % Read mini-batch of data and convert the labels to dummy
                % variables.
                idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
                X = XTrain(:,:,:,idx);
                
                
                Y = YTrain(idx,:);
                Y(find(Y==-1))=0;
                
                
                % Convert mini-batch of data to a dlarray.
                dlX = dlarray(single(X),'SSCB');
                dlX = gpuArray(dlX);
                
                % Evaluate the model gradients and loss using dlfeval and the
                % modelGradients helper function.
                [grad,state,loss] = dlfeval(@modelGraML,dlnet,dlX,Y');
                dlnet.State = state;
                
                % Update the network parameters using the Adam optimizer.
                %                     if id==1
                [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad,averageGrad,averageSqGrad,iteration);
                %                     else
                %                         epsilon = 1e-8;beta2 = 0.999;beta1 = 0.9;lr = 0.001;
                %                         [dlnet,averageGrad,averageSqGrad] = adamupdateStoc(dlnet,grad,averageGrad,averageSqGrad,iteration,lr, beta1, beta2, epsilon,id-1);
                %                     end                %
                % Display the training progress..
                D = duration(0,0,toc(start),'Format','hh:mm:ss');
                addpoints(lineLossTrain,iteration,loss)
                title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end
        end
        
        %creo test set
        clear nome test testImages
        for pattern=1:length(Test)
            IM=Test{pattern};%singola data immagine
            IM=imresize(IM,[siz(1) siz(2)]);
            if size(IM,3)==1
                IM(:,:,2)=IM;
                IM(:,:,3)=IM(:,:,1);
            end
            testImages(:,:,:,pattern)=uint8(IM);
        end
        
        dlXTest = dlarray(single(testImages),'SSCB');
        dlXTest = gpuArray(dlXTest);
        dlYPred = predict(dlnet,dlXTest);
        YPredValidation = single(gather(extractdata(dlYPred) > (1/numClasses*2)));
        
        lgraph = layerGraph(dlnet);
        lgraph=removeLayers(lgraph, {'fc','softmax'});
        NuovoTest = squeeze(predict(dlnetwork(lgraph),dlXTest));
        NuovoTrain = squeeze(predict(dlnetwork(lgraph),dlarray(single(trainingImages),'SSCB')));
        NuovoTest=single(gather(extractdata(NuovoTest)));
        NuovoTrain=single(gather(extractdata(NuovoTrain)));
        
        save(strcat(URI,'ResNetMultilabelSigmoid_',num2str(datas),'_',num2str(fold),'_',num2str(itera),'_',num2str(reitero),'.mat'),'dlYPred','dlnet','ord','NuovoTest','NuovoTrain')
        
    end
end



function [gradients,state,loss] = modelGraML(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);

loss = crossentropy(dlYPred,Y,'TargetCategories','independent');

gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end
