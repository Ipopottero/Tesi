clear train_data3 test_data3

%training set
try
    dlX = dlarray(train_data,'BTC');
    [dlY,dlYprePool,dlYpostPool] = model(dlX,parameters);
    dlY=extractdata(gather(dlY));
    dlYprePool=extractdata(gather(dlYprePool));
    dlYpostPool=extractdata(gather(dlYpostPool));
catch
    dlX = dlarray(train_data(1:ceil(end/2),:),'BTC');
    [dlY,dlYprePool,dlYpostPool] = model(dlX,parameters);
    dlX = dlarray(train_data(1+ceil(end/2):end,:),'BTC');
    [dlYA,dlYprePoolA,dlYpostPoolA] = model(dlX,parameters);
    dlY=[extractdata(gather(dlY)) extractdata(gather(dlYA))];
    dlYprePool=[extractdata(gather(dlYprePoolA)) extractdata(gather(dlYprePoolA))];
    dlYpostPool=[extractdata(gather(dlYpostPoolA)) extractdata(gather(dlYpostPoolA))];
end

% for i=1:size(train_data,1)
%     dlX = dlarray(train_data(i,:),'BTC');
%     [dlY(i,:),dlYprePool(i,:),dlYpostPool(i,:)] = model(dlX,parameters);
% end

train_data1=dlY;
train_data2=dlYpostPool;
% for pattern=1:size(dlYprePool,2)
%     train_data3(pattern,:)=squeeze(reshape(dlYprePool(:,pattern,:),[],1));
% end


%test set
dlX = dlarray(test_data,'BTC');
[dlY,dlYprePool,dlYpostPool] = model(dlX,parameters);
dlY=extractdata(gather(dlY));
dlYprePool=extractdata(gather(dlYprePool));
dlYpostPool=extractdata(gather(dlYpostPool));

test_data1=dlY;
test_data2=dlYpostPool;
% for pattern=1:size(dlYprePool,2)
%     test_data3(pattern,:)=squeeze(reshape(dlYprePool(:,pattern,:),[],1));
% end