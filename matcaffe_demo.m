%  Copyright (c) 2015, Omkar M. Parkhi
%  All rights reserved.

% Both typical matlab functions...


%img_dir = './imgs';
%cd(img_dir);
s = dir('*.jpg');
names = {s.name};
newdim = 224;

for i = 1: 100
    
    old_name = names{i};
    disp(old_name);

    img = single(img);

    % Pre-processing steps...why were these numbers used etc?
    averageImage = [129.1863,104.7624,93.5940] ;

    img = cat(3,img(:,:,1)-averageImage(1),...
        img(:,:,2)-averageImage(2),...
        img(:,:,3)-averageImage(3));
    
    new_name = strcat('proc/', 'cropped_norm_', old_name);
    imwrite(img, new_name);
    
end

