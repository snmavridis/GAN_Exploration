clear; clc; close all;

imageTextFile = 'C:\Users\snmav\Documents\MATLAB\fgvc-aircraft-2013b\data\images_family_trainval.txt';
imageFolder = "C:\Users\snmav\Documents\MATLAB\fgvc-aircraft-2013b\data\images";

fileID = fopen(imageTextFile);
C = textscan(fileID,'%s %s %s');
fclose(fileID);

len = length(C{1,1});


for i = 1:len
    imageLbl(i) = string(C{1,1}{i});
    imageMfg(i) = string(C{1,2}{i});
    imageFam(i) = string(C{1,3}{i});
end
k = 0;
for i = 1:len
    if strcmp(imageFam(i),"737")
        k = k+1;
        ImagesToUse(k) = imageLbl(i);
    end    
end

for i = 1:length(ImagesToUse)
    destinationFolder = 'C:\Users\snmav\Documents\MATLAB\fgvc-aircraft-2013b\data\737';
    A = append(imageFolder,'\',ImagesToUse(i),'.jpg');
    im = imread(A);
    [imHeight, imWidth, imRGB] = size(im);
    imTrim = imHeight*0.97;
    figure(1)
    imshow(im([1:imTrim],:,:))
    saveas(figure(1),append(destinationFolder,'\',ImagesToUse(i),'.png'));
end
