clear;
time = dir('33*');
t = {time(:).name};
for z = 1:length(t)
    d = dir(fullfile(t{z},'/','masks','*.png'));
    p = {d(:).folder};
    n = {d(:).name};
    
    for k = 1 : length(n) 
       
        mask = imread(strcat(p{k},'/',n{k}));   % read the segmented image
        mask_b = im2bw(mask);                   % create binary image 
        skeleton = bwmorph(mask_b, 'skel', Inf);% create initial skeleton
        
        outline = bwmorph(mask_b, 'remove');    % get outline of the binary image
        B = bwmorph(skeleton, 'branchpoints');  % get branchpoints of the skeleton
        E = bwmorph(skeleton, 'endpoints');     % get endpoints of the skeleton
        
        % get coordinates of endpoints and branch roots
        [ye,xe] = find(E);  
        B_loc = find(B);    
        [yb,xb] = find(B);  
        % get total of endpoints and branchpoints in the skeleton
        sum_E = sum(E(:));  
        sum_B = sum(B(:));  
        
        % start from an endpoint, walk through the skeleton, find all pixels 
        % that are closer than the nearest branchpoint, then remove those
        % pixels. 
        % Dmask = isolated branches
        % DDmask = isolated branches + 1 pixel, to keep branches with
        % common roots connected

        Dmask = false(size(skeleton));  
        DDmask = false(size(skeleton));

        for h = 1:sum_E
            D = bwdistgeodesic(skeleton,xe(h),ye(h));   
            distanceToBranchPt = min(D(B_loc));      
            DDmask(D < distanceToBranchPt+1) = true;
            Dmask(D < distanceToBranchPt) = true;     
        end
        
        skelD = skeleton - Dmask; % centerline of skelelton without branches 
        
        % trace the skeleton from the image and plot it
        imshow(skelD)
        hold on
        contour = bwtraceboundary(skelD, [yb(1) xb(1)],'W');
        plot(contour(:,2),contour(:,1),'-w','LineWidth',2); 
    
        % find the mean (x,y) of the endpoints of each component and plot a
        % line from the root to the mean point
        for i = 1:sum_B
            [r,c] = find(bwlabel(DDmask)==i); % label each connected component
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
            plot(X,Y,'.-w','LineWidth',2);
        end

        % get all x,y coordinates of the skeleton and put into a matrix
        data = get(gca, 'Children');
        xd = get(data, 'XData');
        xdata = cell2mat(reshape(xd',1,[]));
        yd = get(data, 'YData'); 
        ydata = cell2mat(reshape(yd',1,[]));
    
        m = [xdata ; ydata];
    
        % save the matrix into a csv file (col 1 = x, col 2 = y)
        path = strcat(t{z},'/skeletons/','matrix', n{k}, '.csv');
        path = erase(path, '.png');
        path = erase(path, 'mask');
        writematrix(m.', path);
    
        hold off
    
    end
end

