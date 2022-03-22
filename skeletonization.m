cd Individual_masks
dc = dir('*.png');
names = {dc(:).name};

for k = 1 : length(names)
    mask = imread(names{k});
    mask2 = im2bw(mask); 
    outline = bwmorph(mask2, 'remove');
    skeleton = bwmorph(mask2,'skel',Inf);
    full = (skeleton + outline);
    path = strcat('/home/keeganfl/Desktop/Math_612_project/my_closet/skel',names{k});
    imwrite(full, path)
end

