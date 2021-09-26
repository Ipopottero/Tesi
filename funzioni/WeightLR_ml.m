function Y=WeightLR_ml(p, g, avg_g, avg_gsq, t, lr, beta1, beta2, epsilon,approccio,gradOld);

%   Input GRAD contains the gradients of the loss with respect to each of
%   the network parameters. Inputs AVG_G and AVG_SQG contain, the moving
%   average of the parameter gradients and the moving average of the
%   element-wise squares of the parameter gradients, respectively.

if approccio==1
    X=abs(g-avg_gsq);%
    X=X./max(X(:));
    X=X*4;
    Y = 1 ./ (1 + exp(-X));
    
elseif approccio==2%cos decay
    drop=.01;
    lri=0.001;
    learningRate=cos(0:pi/(30):pi)*lri;
    for i=1:30
        if learningRate(i)<0
            learningRate(i)=-learningRate(i);
        end
        if learningRate(i)==0
            learningRate(i)=lri*0.9;
        end
        learningRate(i)=learningRate(i)*exp(-drop*i);
    end
    learningRate=learningRate./max(learningRate(:));
    X=abs(g-avg_gsq);%
    X=X./max(X(:));
    X=X*2*(2-learningRate(mod(t,30)+1));
    Y = 1 ./ (1 + exp(-X));
    Y=Y+(1-learningRate(mod(t,30)+1));
    
elseif approccio==3%sin_decay
    drop=.01;
    lri=0.001;
    %sin
    learningRate=sin(0:pi/(30-1):pi)*lri;
    learningRate(1)=learningRate(2);
    learningRate(30)=learningRate(30-1);
    for i=1:30
        learningRate(i)=learningRate(i)*exp(-drop*i);
    end
    learningRate=learningRate./max(learningRate(:));
    X=abs(g-avg_gsq);%
    X=X./max(X(:));
    X=X*2*(2-learningRate(mod(t,30)+1));
    Y = 1 ./ (1 + exp(-X));
    Y=Y+(1-learningRate(mod(t,30)+1));
    
elseif approccio==4
    %cyclic
    drop=.01;
    lri=0.001;
    [learningRate]=createLearningRate('cyc',30,lri);
    learningRate=learningRate./max(learningRate(:));
    X=abs(g-avg_gsq);%
    X=X./max(X(:));
    X=X*2*(2-learningRate(mod(t,30)+1));
    Y = 1 ./ (1 + exp(-X));
    Y=Y+(1-learningRate(mod(t,30)+1));
    
elseif approccio==5%cos decay
    drop=.01;
    lri=0.001;
    learningRate=cos(0:pi/(30):pi)*lri;
    for i=1:30
        if learningRate(i)<0
            learningRate(i)=-learningRate(i);
        end
        if learningRate(i)==0
            learningRate(i)=lri*0.9;
        end
        learningRate(i)=learningRate(i)*exp(-drop*i);
    end
    learningRate=learningRate./max(learningRate(:));
    X=abs(g-avg_gsq);%
    X=X./max(X(:));
    X=X*4*(2-learningRate(mod(t,30)+1));
    Y = 1 ./ (1 + exp(-X));
    
    
    
    %xe^-x
elseif approccio==10
    m=1.50;
    p=2;
    X=abs(g-avg_gsq);
    X=(X.*exp(-(X)*p));
    X=m*X./max(X(:));
    Y=X;
    
    %Sto
elseif approccio==25
    m=1.50;
    p=1;
    q=4;
    X=abs(g-avg_gsq);
    X=(q*(X).*exp(-(X)*p*q));
    R=rand(size(X))+0.5;
    X=X.*R;
    X=m*X./max(X(:));
    Y=X;
    
end

if sum(isnan(Y(:)))>0
    for a=1:size(Y,1)
        for b=1:size(Y,2)
            if isnan(Y(a,b))
                Y(a,b)=1;
            end
        end
    end
else
    keyboard
end