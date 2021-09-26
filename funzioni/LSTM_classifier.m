

for reitero=1:3
    if reitero==1
        numEpochs = 20;
    elseif reitero==2
        numEpochs = 50;
    elseif reitero==3
        numEpochs = 100;
    end
    for itero=1:10
        
        TR=train_data;
        y=train_target;
        TE=test_data;
        yy=test_target;
        
        %% LSTM con soglia %%
        %duplico pattern per ogni sua classe
        clear label
        t=1;
        NTR=TR;
        for i=1:size(TR,1)
            %quante label ha il dato pattern?
            classi=find(y(i,:)==1);
            if length(classi)>1
                for cl=1:length(classi)
                    NTR(t,:)=TR(i,:);
                    label(t)=classi(cl);
                    t=t+1;
                end
            else
                NTR(t,:)=TR(i,:);
                label(t)=classi;
                t=t+1;
            end
        end
        TR=NTR;
        clear NTR
        y=label;
        
        inputSize = size(train_data,2);
        numHiddenUnits = 100;
        numClasses = size(train_target,2);
        layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
        
        
        
        LR=0.001;
        maxEpochs = 100;
        miniBatchSize = 27;
        options = trainingOptions('adam', ...
            'GradientThreshold',1, ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'SequenceLength','longest', ...
            'Shuffle','never', ...
            'Verbose',0, ...
            'Plots','training-progress');
        
        clear Xtrain NuovoTrain
        for i=1:size(TR,1)
            Xtrain{i}=TR(i,:)';
        end
        net = trainNetwork(Xtrain,categorical(y)',layers,options);

        for i=1:size(TR,1)
            NuovoTrain(i,:)=activations(net,Xtrain{i}, 'biLSTM');
        end
        
        clear Xtest NuovoTest
        for i=1:size(TE,1)
            Xtest{i}=TE(i,:)';
            NuovoTest(i,:) = activations(net,Xtest{i}, 'biLSTM');
        end
        scoreLSTM = predict(net,Xtest, ...
            'MiniBatchSize',miniBatchSize, ...
            'SequenceLength','longest');
        

%         YPredValidation = single(scoreLSTM > (1/numClasses*2) );
%         if min(test_target(:))==-1;
%             YPredValidation(find(YPredValidation==0))=-1;
%         end
%         res(1:5) = Evaluation(YPredValidation,scoreLSTM,test_target);
        
        save(strcat(URI,'LSTMor_',num2str(datas),'_',num2str(fold),'_',num2str(itero),'_',num2str(reitero),'.mat'),'net','scoreLSTM','NuovoTest','NuovoTrain')
        
        
    end
end
