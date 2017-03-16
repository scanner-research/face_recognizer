    
disp('starting the cropping stuff!');
% 224 is the expected dimension?
extend = 0.1;
newdim = 224;

old_name = './imgs/aj';
img = imread(strcat(old_name, '.jpeg'));

size(img);
box = [100, 0, 250, 140];

%disp(length(box));

%make square
width = round(box(3)-box(1));
height = round(box(4)-box(2));

length = (width + height)/2;

% centrepoint?!!! as compared to subtracting averages?
centrepoint = [round(box(1)) + width/2 round(box(2)) + height/2];
x1= centrepoint(1) - round((1+extend)*length/2);
y1= centrepoint(2) - round((1+extend)*length/2);
x2= centrepoint(1) + round((1+extend)*length/2);
y2= centrepoint(2) + round((1+extend)*length/2);


% prevent going off the page
x1= max(1,x1);
y1= max(1,y1);
x2= min(x2,size(img,2));
y2= min(y2,size(img,1));

% does the cropping
img = img(y1:y2,x1:x2,:);
sizeimg = size(img)

% doesn't work to get newdim X newdim if width and height are different.
img = imresize(img,(newdim/sizeimg(1)));

disp(size(img));
%img = single(img);
% Pre-processing steps...why were these numbers used etc?
%averageImage = [129.1863,104.7624,93.5940] ;

%img = cat(3,img(:,:,1)-averageImage(1),...
%    img(:,:,2)-averageImage(2),...
%    img(:,:,3)-averageImage(3));


new_name = strcat(old_name, '_face_crop.png');
imwrite(img, new_name);
