
%Uso i 5 “numeric”  di D:\c\Lavoro\TOOL\IMCC
%sono già splittati fra training e test set
if datas==1
    load('cal500.mat')
    NF=1;
elseif datas==2
    load('image.mat')
    NF=1;
elseif datas==3
    load('scene.mat')
    NF=1;
elseif datas==4
    load('yeast.mat')
    NF=1;
elseif datas==5
    load('arts.mat')
    NF=1;
    
    %due di ATC
elseif datas==6
    load ATC_42_3883.mat
    NF=10;
    atc_fea=atc_fea';
    atcClass=atcClass';
    
elseif datas==7
    load('D:\c\Lavoro\Implementazioni\iATCmatricizzazione\FusioneATC_NR_FR.mat','atc_fea')
    load('ATC_42_3883.mat','atcClass');
    NF=10;
    atc_fea=atc_fea';
    atcClass=atcClass';
    
    %D:\c\Lavoro\TOOL\Predicting drug side effects\Liu dataset and experiments
elseif datas==8
    load('Liu_dataset')
    clear X
    X{1}=Enzymes;
    X{2}=Pathways;
    X{3}=Targets;
    X{4}=Transporters;
    X{5}=Treatment;
    X{6}=chemical;
    Y=side_effect;
    NF=5;%5-fold cross validation
    atc_fea=[X{1} X{2} X{3} X{4} X{5} X{6}];
    atcClass=Y;
    
    % D:\c\Lavoro\DATA\DATA\mAnimal   sequenza amino-acidica
elseif datas==9
    clear atc_fea
    load('D:\c\Lavoro\DATA\DATA\mAnimal\dataSupp2.mat')
    Y=zeros(3919,20)-1;
    for i=1:length(DATA{2})
        Y(i,DATA{2}{i})=1;
        if sum(Y(i,:)==1)==0
            i
        end
    end
    load('D:\c\Lavoro\DATA\DATA\mAnimal\dataSupp3.mat')
    for i=1:length(DATA{2})
        atc_fea(i,:)=DATA{2}{i};
    end
    atcClass=Y;
    NF=10;

elseif datas==10
    
    
    
    
end