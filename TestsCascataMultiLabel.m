clear all
close all force
warning off


URI='C:\Users\matti\Desktop\Tesi\scores\'; %dove salvo scores

% %per tutti i datasets lancio tutti i classificatori
for datas=1 %:5
    LeggiDataMultiLabel;%leggo il dataset
    close all force

    %per tutti i fold
    for fold=1

        %supporta solo 1 fold
        [NF, test_data, test_target, train_data, train_target] = load_data(datas);

%         load('DatiCheUso.mat','DatasetUsati')
%         train_data=DatasetUsati{datas}{fold}{1};
%         test_data=DatasetUsati{datas}{fold}{2};
%         train_target=DatasetUsati{datas}{fold}{3};
%         test_target=DatasetUsati{datas}{fold}{4};

        %controllo se vettore label sia n° pattern x n° classi
        if size(train_target,1)==size(train_data,1)
        else
            train_target=train_target';
            test_target=test_target';
        end
        
        % Rimuovo i null
        tot=find(sum(train_data)==0);
        train_data(:,tot)=[];
        test_data(:,tot)=[];


        if datas==5 || datas==8 %PCA
            [coeff,scoreTrain,~,~,explained,mu] = pca(train_data);
            sum_explained = 0;
            idx = 0;
            while sum_explained < 99
                idx = idx + 1;
                sum_explained = sum_explained + explained(idx);
            end
            train_data=scoreTrain(:,1:idx);
            test_data= (test_data-mu)*coeff(:,1:idx);
        end
        
        %porto tutti i target in {-1,1}
        train_target(find(train_target==0))=-1;
        test_target(find(test_target==0))=-1;

%fine preprocessing

        %lancio in cascata i vari classificatori
        GRUclassifierVarianteAdamEnsembleHYBRID; 

    end


end

