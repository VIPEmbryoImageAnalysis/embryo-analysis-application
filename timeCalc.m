function tB1 = timeCalc(imagename)
roi = [435 472 45 15]; 
A = '0123456789';
M = [];

%I = imread(imagename);
% I = imcrop(I,roi);
I = imcrop(imagename,roi);
I = imsharpen(I);
I = imbinarize(I);
I = imresize(I,1.3);
I = bwmorph(I, 'thicken', 3);
I = bwmorph(I, 'bothat', 1);
I = bwmorph(I, 'tophat', 1);
results = ocr(I,'CharacterSet', '.0123456789h', 'TextLayout','Block');
m = results.Text;
for r = 1:length(m)
    if ismember(m(1,r),A)
        M = [M m(1,r)];
    end
end
M1 = str2double(M);
tB1 = M1/10;
 end