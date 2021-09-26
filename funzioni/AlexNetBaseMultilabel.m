close all force

gpuDevice(index)

Nepoc=150;
miniBatchSize = 30;
LR=0.001;
net = alexnet;  %load AlexNet
siz=[227 227];
netName = 'alexnet';
inputLayer = net.Layers(1);
siz = inputLayer.InputSize(1:2);




%reshape vector -> matrix
for itera=1:10
    
    TR=  train_data;
    TE=  test_data;
    y=  train_target;
    yy=  test_target;

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
    
    
    %per tutti i pattern con più label inserisco nel training più
    %volte, considerando label diverse
    clear label
    t=1; 
    NTR=trainingImages;
    for i=1:size(y,2)
        %quante label ha il dato pattern?
        classi=find(y(i,:)==1);
        if length(classi)>1
            for cl=1:length(classi)
                NTR(:,:,:,t)=trainingImages(:,:,:,i);
                label(t)=classi(cl);
                t=t+1;
            end
        else
            NTR(:,:,:,t)=trainingImages(:,:,:,i);
            label(t)=classi;
            t=t+1;
        end
    end
    trainingImages=NTR;
    clear NTR
    y=label;
    numClasses = max(y(:));
    %CNN non è multilabel, tanto lo uso come extractor, rendo singola
    %la label
    
    imageSize=size(IM);
    
    trainingImages = augmentedImageSource(imageSize,trainingImages,categorical(y'));
    
    options = trainingOptions('sgdm',...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',Nepoc,...
        'InitialLearnRate',LR,...
        'Verbose',false,...
        'Plots','training-progress');
    
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    try netTransfer = trainNetwork(trainingImages,layers,options);
    catch
        return
    end
    
    clear score
    for i=1:length(Test)
        IM=Test{i};
        IM=imresize(IM,[siz(1) siz(2)]);%si deve fare size immagini per rendere compatibili con CNN
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        [outclass, score(i,:)]= classify(netTransfer,IM);
    end
    
    YPredValidation = single(score > (1/numClasses*2));
%     clear res
%     if min(test_target(:))==-1;
%         YPredValidation(find(YPredValidation==0))=-1;
%     end
%     res(1:5) = Evaluation(YPredValidation',score',test_target');
    
    save(strcat(URI,'AlexNetBase_',num2str(datas),'_',num2str(fold),'_',num2str(itera),'.mat'),'score','netTransfer')
        
end

