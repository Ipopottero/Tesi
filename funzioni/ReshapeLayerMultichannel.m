function matrix = ReshapeLayerMultichannel(array, width, height)
%INPUT
%array: feature vector to reshape
%width, height: size of the output image

%OUTPUT
%matrix -> "image" that can be used to feed CNN

MS=ceil((length(array)/3)^0.5);

R(1:MS^2)=0;
G(1:MS^2)=0;
B(1:MS^2)=0;

pt1=floor(length(array)/3);
pt2=floor(length(array)*(2/3));

R(1:pt1)=array(1:pt1);
G(1:pt2-pt1)=array(pt1+1:pt2);
B(1:length(array)-pt2)=array(pt2+1:length(array));


R=reshape(R,MS,MS);
G=reshape(G,MS,MS);
B=reshape(B,MS,MS);

matrix(:,:,1)=R;
matrix(:,:,2)=G;
matrix(:,:,3)=B;

matrix = imresize(matrix,[width height]); 

end