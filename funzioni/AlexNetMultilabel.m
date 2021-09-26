clear all
close all force

gpuDevice(1)

gradDecay = 0.75;
sqGradDecay = 0.95;

% Load training data and encode.
datas=8;
% %carica dataset
try load(strcat('DatasColor_',int2str(datas)),'DATA');
catch
    load(strcat('Datas_',int2str(datas)),'DATA');
end

NF=size(DATA{3},1); %number of folds
DIV=DATA{3};%divisione fra training e test set
DIM1=DATA{4};%numero di training pattern
DIM2=DATA{5};%numero di pattern
yE=DATA{2};%label dei patterns
NX=DATA{1};%immagini
DA=DATA{4};
classes = categories(categorical(yE));
numClasses = numel(classes);



net = alexnet;
lgraph = layerGraph(net);
%lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');
lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    ];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'avg_pool','fc');
inputLayer = net.Layers(1);
siz = inputLayer.InputSize(1:2);


layersTransfer = net.Layers(1:end-3);
        layers = [
            layersTransfer
            fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
            softmaxLayer
            ];

%run stocastic
% for metodo=[5 6 8]
%     for itera=1:22
%         for fold=1:5
%             clear Training Test
%             TR=data(idTR{fold},:);
%             TE=data(idTE{fold},:);
%             
%             %random factors for AVG and SUM
%             DimDCT=zeros(size(TR,2),size(TR,2));
%             OrdDct=zeros(size(TR,2),size(TR,2),6);
%             for ii=1:size(TR,2)
%                 for jj=1:size(TR,2)
%                     %random subset of 2-6 features
%                     [~, n]=min(rand(min([5 size(TR,2)-1]),1));n=n+1;
%                     [a,b]=sort(rand(size(TR,2),1));
%                     DimDCT(ii,jj)=n;%number of random features
%                     OrdDct(ii,jj,1:n)=b(1:n);%id of the random selected features
%                 end
%             end
%             
%             
%             for pattern=1:size(TR,1)
%                 network_features=TR(pattern,:);
%                 if pattern==1 && fold==1
%                     ord=randperm(length(network_features));
%                 end
%                 if mod(itera,2)==1 && itera<=15
%                     % Creare immagine riarrangiando in maniera random
%                     NW=ceil(length(network_features)^0.5);
%                     NR(1:NW^2)=0;
%                     NR(1:length(network_features))=network_features(ord);
%                     IM=reshape(NR,NW,NW);
%                     if size(IM,3)==1
%                         IM(:,:,2)=IM;
%                         IM(:,:,3)=IM(:,:,1);
%                     end
%                     %reshape multichannel
%                 elseif mod(itera,2)==0 && itera<=15
%                     NW=ceil(length(network_features)^0.5);
%                     NR(1:NW^2)=0;
%                     NR(1:length(network_features))=network_features(ord);
%                     IM = ReshapeLayerMultichannel(NR, NW, NW);
%                 elseif itera>15
%                     %% method named SUM in the paper
%                     I=network_features;
%                     IM=zeros(length(I),length(I));
%                     for ii=1:length(I)%I is a given image
%                         for jj=1:length(I)
%                             %random subset of 2-6 features
%                             n=DimDCT(ii,jj);
%                             %to reduce the subset to a single dimension
%                             D=sum(I(OrdDct(ii,jj,1:n)));
%                             IM(ii,jj)=D(1);
%                         end
%                     end
%                     IM(:,:,2)=IM;
%                     IM(:,:,3)=IM(:,:,1);
%                     IM=imresize(IM,[250 250]);
%                     
%                 end
%                 Training{pattern}=IM;
%             end
%             
%             for pattern=1:size(TE,1)
%                 network_features=TE(pattern,:);
%                 if mod(itera,2)==1 && itera<=15
%                     % Creare immagine riarrangiando in maniera random
%                     NW=ceil(length(network_features)^0.5);
%                     NR(1:NW^2)=0;
%                     NR(1:length(network_features))=network_features(ord);
%                     IM=reshape(NR,NW,NW);
%                     if size(IM,3)==1
%                         IM(:,:,2)=IM;
%                         IM(:,:,3)=IM(:,:,1);
%                     end
%                     %reshape multichannel
%                 elseif mod(itera,2)==0 && itera<=15
%                     NW=ceil(length(network_features)^0.5);
%                     NR(1:NW^2)=0;
%                     NR(1:length(network_features))=network_features(ord);
%                     IM = ReshapeLayerMultichannel(NR, NW, NW);
%                 elseif itera>15
%                     %% method named SUM in the paper
%                     I=network_features;
%                     IM=zeros(length(I),length(I));
%                     for ii=1:length(I)%I is a given image
%                         for jj=1:length(I)
%                             %random subset of 2-6 features
%                             n=DimDCT(ii,jj);
%                             %to reduce the subset to a single dimension
%                             D=sum(I(OrdDct(ii,jj,1:n)));
%                             IM(ii,jj)=D(1);
%                         end
%                     end
%                     IM(:,:,2)=IM;
%                     IM(:,:,3)=IM(:,:,1);
%                     IM=imresize(IM,[250 250]);
%                     
%                 end
%                 Test{pattern}=IM;
%             end
%             
%             %run stocastic
%             TestRandomReLu2020transferLearning(Training, Test, labelTR{fold} , labelTE{fold} , fold, 'DTI', metodo, 0.001, 30, [],255,itera,1, index)
%             
%             %ora lanciare baseline con ensemble CNN base
%             BaselineCNN_fold(Training, Test, labelTR{fold} , labelTE{fold} , fold, 'DTI', metodo, 0.001, 30, 'ReLu',1,itera, index)
%             
%         end
%     end
% end



for fold=1:NF
    close all force
    
    try
        DIM1=DA(fold);
    end
    
    trainPattern=(DIV(fold,1:DIM1));
    testPattern=(DIV(fold,DIM1+1:DIM2));
    y=yE(DIV(fold,1:DIM1));
    yy=yE(DIV(fold,DIM1+1:DIM2));
    YTrain=categorical(y);
    
    %creo il training set
    clear nome trainingImages
    trainingImages = zeros(siz(1),siz(2),3,length(y)*4,'uint8');
    tmp=1;
    for pattern=1:length(y)
        IM=NX{DIV(fold,pattern)};%singola data immagine
        
        IM=imresize(IM,[siz(1) siz(2)]);%si deve fare size immagini per rendere compatibili con CNN
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        
        trainingImages(:,:,:,tmp)=IM; label(tmp)=y(pattern);tmp=tmp+1;
        %data augmentation
        trainingImages(:,:,:,tmp)=flip(IM,2); label(tmp)=y(pattern);tmp=tmp+1;
        trainingImages(:,:,:,tmp)=flip(IM,1); label(tmp)=y(pattern);tmp=tmp+1;
        scale=rand(1,2)+1;
        trainingImages(:,:,:,tmp)=imresize(imresize(IM,scale(1)),siz(1:2)); label(tmp)=y(pattern);tmp=tmp+1;

    end
    imageSize=size(IM);
    y=label;
    YTrain=categorical(y);
    XTrain = single(trainingImages);
    
    for reitero=1:7%così poi controllo ensemble
        for id=1:7
            
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
            numEpochs = 20;
            numObservations = numel(y);
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
                idx = randperm(numel(YTrain));
                XTrain = XTrain(:,:,:,idx);
                YTrain = YTrain(idx);
                
                for i = 1:numIterationsPerEpoch
                    iteration = iteration + 1;
                    
                    % Read mini-batch of data and convert the labels to dummy
                    % variables.
                    idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
                    X = XTrain(:,:,:,idx);
                    
                    
                    Y = zeros(numClasses, miniBatchSize, 'single');
                    for c = 1:numClasses
                        Y(c,YTrain(idx)==classes(c)) = 1;
                    end
                    
                    % Convert mini-batch of data to a dlarray.
                    dlX = dlarray(single(X),'SSCB');
                    dlX = gpuArray(dlX);
                    
                    % Evaluate the model gradients and loss using dlfeval and the
                    % modelGradients helper function.
                    [grad,state,loss] = dlfeval(@modelGra,dlnet,dlX,Y);
                    dlnet.State = state;
                    
                    % Update the network parameters using the Adam optimizer.
                    if id==1
                        [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad,averageGrad,averageSqGrad,iteration);
                    else
                        epsilon = 1e-8;beta2 = 0.999;beta1 = 0.9;lr = 0.001;
                        [dlnet,averageGrad,averageSqGrad] = adamupdateStoc(dlnet,grad,averageGrad,averageSqGrad,iteration,lr, beta1, beta2, epsilon,id-1);
                    end                %
                    % Display the training progress..
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    addpoints(lineLossTrain,iteration,loss)
                    title("Epoch: " + epoch + ", Elapsed: " + string(D))
                    drawnow
                end
            end
            
            
            
            %creo test set
            clear nome test testImages
            for pattern=ceil(DIM1)+1:ceil(DIM2)
                IM=NX{DIV(fold,pattern)};%singola data immagine
                IM=imresize(IM,[siz(1) siz(2)]);
                if size(IM,3)==1
                    IM(:,:,2)=IM;
                    IM(:,:,3)=IM(:,:,1);
                end
                testImages(:,:,:,pattern-ceil(DIM1))=uint8(IM);
            end
            
            dlXTest = dlarray(single(testImages),'SSCB');
            dlXTest = gpuArray(dlXTest);
            dlYPred{fold}{id} = predict(dlnet,dlXTest);
            [~,idx] = max(extractdata(dlYPred{fold}{id} ),[],1);
            YPred = classes(idx);
            YTest=categorical(yy);
            Perf(id,fold) = mean(YPred==YTest')
        end
        
        
        [~,idx] = max(extractdata(dlYPred{fold}{1} )+extractdata(dlYPred{fold}{2}) +extractdata(dlYPred{fold}{3}),[],1);
        YPred = classes(idx);
        Perf(id+1,fold) = mean(YPred==YTest')
        
        [~,idx] = max(extractdata(dlYPred{fold}{1} )+extractdata(dlYPred{fold}{3} )+extractdata(dlYPred{fold}{2} )+extractdata(dlYPred{fold}{4} ),[],1);
        YPred = classes(idx);
        Perf(id+2,fold) = mean(YPred==YTest')
        
        [~,idx] = max(extractdata(dlYPred{fold}{1} )+extractdata(dlYPred{fold}{3} )+extractdata(dlYPred{fold}{2} )+extractdata(dlYPred{fold}{4} ) +extractdata(dlYPred{fold}{5} ),[],1);
        YPred = classes(idx);
        Perf(id+3,fold) = mean(YPred==YTest')
        
        ScoreIterazione{reitero}=dlYPred;
        save(strcat('D:\c\Lavoro\Implementazioni\DeepFeatures\AdamScores\AdamScore_',num2str(datas),'_',num2str(id),'_',num2str(reitero),'.mat'),'dlYPred','ScoreIterazione')
        
    end
end




function [gradients,state,loss] = modelGra(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);

loss = crossentropy(dlYPred,Y,'TargetCategories','independent');

gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end
