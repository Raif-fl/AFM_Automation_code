clear;

d = dir('*.png');
n = {d(:).name};

for k = 1 : length(n)

    mask = imread(n{k});
    mask_b = im2bw(mask);
    skeleton = bwmorph(mask_b, 'skel', Inf);
    
    outline = bwmorph(mask_b, 'remove');
    B = bwmorph(skeleton, 'branchpoints');
    E = bwmorph(skeleton, 'endpoints');
    
    [ye,xe] = find(E);
    B_loc = find(B);
    [yb,xb] = find(B);
    sum_E = sum(E(:));
    sum_B = sum(B(:));
    
    Dmask = false(size(skeleton));
    dd = false(size(skeleton));
        
    for h = 1:sum_E
        D = bwdistgeodesic(skeleton,xe(h),ye(h));
        distanceToBranchPt = min(D(B_loc));
        dd(D < distanceToBranchPt+1) = true; %skeleton of isolated branches
        Dmask(D < distanceToBranchPt) =true;
    end
    
    skelD = skeleton - Dmask;

    cont = skelD - skeleton;
    
    imshow(cont)
    hold on
    contour = bwtraceboundary(skelD, [yb(1) xb(1)],'NW');
    plot(contour(:,2),contour(:,1),'.-w','LineWidth',2);


    for i = 1:sum_B
        [r,c] = find(bwlabel(dd)==i);
        sum_x = 0;
        sum_y = 0;
        ind = 0;
    
            for j = 1:sum_E 
                if ismember([xe(j), ye(j)],[r,c]) 
                    ind = ind+1;
                    sum_x = sum_x+xe(j);
                    xmean = sum_x/ind;
                    sum_y = sum_y+ye(j);
                    ymean = sum_y/ind;
                end 
            end
            hold on
            X = [xmean,xb(i)] ;
            Y = [ymean,yb(i)] ;
            plot(X,Y,'.-w','LineWidth',2)
    end

    full_path = strcat('skel',n{k});
    
    saveas(gcf,full_path)
    hold off


end



