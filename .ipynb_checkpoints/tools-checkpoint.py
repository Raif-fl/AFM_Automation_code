import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from PIL import Image
import cv2
import pandas as pd
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
import re
import imageio
from fil_finder import FilFinder2D
from astropy import units as u
from radfil import radfil_class, styles
from shapely.geometry import Polygon
from collections import OrderedDict

def atoi(text):
    '''
    A function which automatically converts text to integers. 
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    A function which allows input files to be sorted by timepoint.
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_boxes(outl):
    '''
    A function used to compute the bounding boxes for all of the outlines from
    a single image. 
    '''
    # Initialize a list of bounding boxes. 
    boxes = []
    for o in outl:
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(o)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order
        box = perspective.order_points(box)
        # append the bounding box to the list.
        boxes.append(box)
    return boxes

def get_overlap(outl_list, boxes_list):
    '''
    A function which iterates over all cell outlines in an image and detects how much of the cell's
    surface is overlapping/touching another cell's. 
    
    Parameters
    ---------------
    outl: list of 2D arrays
    List should include one outline for each cell in image and arrays should 
    contain the coordinates of the outlines. 
    
    boxes: list of 2D arrays 
    list should include one box for each cell in image and arrays should 
    contain the coordinates of the bounding box corners. 
    
    Returns
    ---------------
    cell_overlaps: list of integers
    list should contain an integer for each cell in the original image which
    specifies the number of pixels in the cell outline that are touching
    another cell outline. 
    
    '''
    overl_list = []
    for outl, boxes in zip(outl_list, boxes_list):
        # initialize a list to save the overlaps
        cell_overlaps = []
        # Loop through each cell in the image. 
        for out_cell, box_cell in zip(outl, boxes):
            adj = 0
            # For each cell in the image, loop through everyother cell in the image.
            for out_oth, box_oth in zip(outl, boxes):
                # detect if the two cells have overlapping bounding boxes.
                # If bounding boxes do not overlap, skip to next cell. 
                p1 = Polygon(box_cell)
                p2 = Polygon(box_oth)
                if p1.intersects(p2) == False or p1 == p2:
                    continue
                # For each pixel in cell outline, measure the distance to the pixels
                # in the other cell outline. If said distance is zero, then the pixels
                # are overlapping. 
                for pixel1 in out_cell:
                    for pixel2 in out_oth:
                        distx = np.abs(pixel1[0]-pixel2[0])
                        disty = np.abs(pixel1[1]-pixel2[1])
                        if distx + disty == 1:
                            adj = adj + 1
            # append the overlap information to our list.
            cell_overlaps.append(adj)
        overl_list.append(cell_overlaps)
    return overl_list 

def peri_area(outl_list):
    '''
    An iterative function that calculates the perimeter and area for all cell outlines within a 
    list of images. 
    
    Parameters
    ---------------
    outl: list of 2D arrays
    A list which contains one outline for each cell in image and arrays should 
    contain the coordinates of the outlines. 
    
    Returns
    ---------------
    peris: list of floats
    List should contain the perimeter size of a cell in pixels for each cell in image.
    
    areas: list of floats
    List should contain the area of a cell in pixels for each cell in image.
    '''
    # Initialize lists to hold the perimeter and area infromation within each individual image. 
    per_list = []
    area_list = []
    for outl in outl_list:
        peris = []
        areas = []
        for o in outl:
            # Calculate the perimeter and area for each outline and 
            # append values to lists. 
            peris.append(cv2.arcLength(o, True))
            areas.append(cv2.contourArea(o))
        per_list.append(peris)
        area_list.append(areas)
    return per_list, area_list

def extract_ind_cells(IDs_list, outl_list, height_img_list, stiff_img_list, pfe_img_list):
    '''
    A function which takes a set of height, stiffness, and peak force error images with associated 
    timepoints, cell outlines, and cell IDs and then isolates individual cell images and masks.
    These isolated cells are rotated so that their longest side is facing downward and are
    then returned as a list. 

    Parameters
    ---------------
    IDs_list: list
    A nested list that contains a list of cell IDs for each image. 
    
    outl_list: list
    A nested list that contains a list of outlines for each cell in each image. 
    
    height_img_list: list 
    a list of height data images. 
    
    stiff_img_list: list 
    a list of stiffness data images. 
    
    pfe_img_list: list 
    a list of peak force error data images. 
    '''
    # Iterate over every image and its associated time point, set of cell ids, and set of cell outlines.
    ind_cell_height_list = []
    ind_cell_stiff_list = []
    ind_cell_pfe_list = [] 
    ind_cell_mask_list = []
    for ID_set, outl, h_img, s_img, p_img in zip(IDs_list, outl_list, height_img_list, stiff_img_list, pfe_img_list):
        ind_cell_height = []
        ind_cell_stiff = []
        ind_cell_pfe = [] 
        ind_cell_mask = []
        # Iterate over each cell in each image. 
        for idx, cell in zip(ID_set, outl):
            # extract the mask from the
            mask = np.zeros(h_img.shape, dtype=np.uint8)
            channel_count = h_img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(mask, [cell], ignore_mask_color)
            
            # Use the cell outline to make a new bounding box.
            rect = cv2.minAreaRect(cell)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Determine the height and width of the bounding box.
            width = int(rect[1][0])
            height = int(rect[1][1])

            # Cut out the part of the image surrounded by the bounding box 
            # and then rotate the image to be at a right angle for each image type.
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            out_height = cv2.warpPerspective(h_img, M, (width, height))
            out_height = cv2.copyMakeBorder(out_height, top = 5, bottom = 5, left = 5, right = 5,
                                            borderType=cv2.BORDER_CONSTANT, value = [0])
            out_mask = cv2.warpPerspective(mask, M, (width, height))
            out_mask = cv2.copyMakeBorder(out_mask, top = 5, bottom = 5, left = 5, right = 5,
                                            borderType=cv2.BORDER_CONSTANT, value = [0])
            # It is necessary to allow for pfe or stiffness images to be not present while height images are present. 
            if np.isnan(s_img.any()) == False:
                out_stiff = cv2.warpPerspective(s_img, M, (width, height)) 
                out_stiff = cv2.copyMakeBorder(out_stiff, top = 5, bottom = 5, left = 5, right = 5,
                                                borderType=cv2.BORDER_CONSTANT, value = [0])
            else:
                out_stiff = np.nan
                
            if np.isnan(p_img.any()) == False:
                out_pfe = cv2.warpPerspective(p_img, M, (width, height))
                out_stiff = cv2.copyMakeBorder(out_stiff, top = 5, bottom = 5, left = 5, right = 5,
                                               borderType=cv2.BORDER_CONSTANT, value = [0])
            else:
                out_pfe = np.nan 
                
            # If the height is greater than the width, rotate it so that the long side is horizontal. 
            if height > width:
                out_height = cv2.transpose(out_height)
                out_mask = cv2.transpose(out_mask) 
                out_stiff = cv2.transpose(out_stiff) 
                out_pfe = cv2.transpose(out_pfe) 
            
            # Append everything into new nested lists. 
            ind_cell_height.append(out_height)
            ind_cell_stiff.append(out_stiff)
            ind_cell_pfe.append(out_pfe)
            ind_cell_mask.append(out_mask)
        ind_cell_height_list.append(ind_cell_height)
        ind_cell_stiff_list.append(ind_cell_stiff)
        ind_cell_pfe_list.append(ind_cell_pfe)
        ind_cell_mask_list.append(ind_cell_mask)
    return(ind_cell_height_list, ind_cell_stiff_list, ind_cell_pfe_list, ind_cell_mask_list)
    
def find_nearest(array, value):
    '''
    Finds the nearest value in an array and returns the index of that value.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def apply_radfil(image_list, mask_list, skel_list, conv = 1.001):
    '''
    This function uses the radfil package to take vertical cuts all along the skeletons created in the previous step. 
    It then collects data from the resulting radfil objects to learn the pixel intensity along the cuts and along the skeleton,
    the width of the cell along the skeleton, and the distance of each datapoint to the skeleton. 
    
    Parameters
    -------------
    image_list: list
    A list of images of individual cells. 
    
    mask_list: list
    A list of masks which cover the individual cells in image_list
    
    skel_list: list
    A list of skeletons which belong to the cells in image list. 
    
    conv: float
    A number which is used to convert from pixel intensity to meaningful units. 
    '''
    int_list = []
    dist_list = []
    width_list = []
    ridge_list = []
    for image_set, mask_set, skel_set in zip(image_list, mask_list, skel_list):
        int_set = []
        dist_set = []
        width_set = []
        ridge_set = []
        for image, mask, skel in zip(image_set, mask_set, skel_set):
            # Load the image, mask, and skeleton
            fil_image = image[:,:,0]
            fil_mask = mask[:,:,0]>0
            fil_skeleton = skel>0
            
            # It is required that the skeleton be slightly truncated to avoid errors.
            # More elegant solutions may include expanding the image boundaries by 1 or two pixels, but it may be a mask problem. 
            fil_skeleton[0,:] = False
            fil_skeleton[:,0] = False
            fil_skeleton[-1,:] = False
            fil_skeleton[:,-1] = False
            
            # Use RadFil to create a radial profile object. 
            radobj=radfil_class.radfil(fil_image, mask=fil_mask, filspine=fil_skeleton, distance=200)
            radobj.build_profile(samp_int=1, shift = False)
            
            # Create lists to hold the information from individual cuts. 
            intensity = []
            ridge = []
            for prof, dist in zip(radobj.dictionary_cuts["profile"], radobj.dictionary_cuts["distance"]):
                # Convert all intensity values to meaningful units. 
                intensity.append(prof*conv)
                # Calculate the converted pixel intensities along the skeleton. 
                ind = find_nearest(dist,0)
                ridge.append(prof[ind]*conv)
             
            # Save the converted intensity and ridgeline intensity values.
            ridge_set.append(ridge)
            int_set.append(intensity)
            
            # save the distances (important for accurate spacing).
            dist_set.append(radobj.dictionary_cuts["distance"])
            
            # save the widths of each cut to get the cell width along the ridgeline.
            width_set.append(radobj.dictionary_cuts["mask_width"])
            
        int_list.append(int_set)
        ridge_list.append(ridge_set)
        dist_list.append(dist_set)
        width_list.append(width_set)
    return(int_list, ridge_list, dist_list, width_list)

def get_max_ID(IDs_list):
    '''
    A function which finds the final cell ID number. 
    '''
    maxi = []
    for ID_set in IDs_list:
        maxi.append(max(ID_set))
    return max(maxi)

def get_metadata(metadata_dict, structural_dict, exact_ID):
    '''
    Parameters
    ---------------  
    metadata_dict: dictionary
    A dictionary which contains all of the metadata that we want to collect for each cell. Note that the keys of this dictionary will
    be used as the names for the columns. 
    
    structural_dict: dictionary
    a dictionary which contains the time point for each image along with the cell IDs for each image. 
    
    exact_ID: Int
    An integer which specifies the cell ID which we would like to extract a metadata table for. 
    
    Returns
    ---------------
    df: pandas dataframe
    A dataframe with parameters as columns (perimeter, area, etc) and timepoints as rows. 
    '''
    df = pd.DataFrame(structural_dict["Time"], columns = ["Time"])
    for key in metadata_dict.keys():
        data = []
        for ID_set, value_set in zip(structural_dict["IDs"], metadata_dict[key]):
            if exact_ID in ID_set:
                index = np.where(np.array(ID_set) == exact_ID)[0][0]
                data.append(value_set[index])
            else:
                data.append(np.nan)
        df[key] = data
    return(df)

# import the necessary packages
class CentroidTracker():
    def __init__(self, maxDisappeared=0):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
    def register(self, rects, centroid, outline):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = (rects, centroid, outline)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
    def update(self, rects, outls):
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, box) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int(np.average(box[:, 0]))
            cY = int(np.average(box[:, 1]))
            inputCentroids[i] = (cX, cY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for r, i, o in zip(rects, range(0, len(inputCentroids)), outls):
                self.register(r, inputCentroids[i], o)
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectinfo = list(self.objects.values())
            objectCentroids = []
            for info in objectinfo:
                objectCentroids.append(info[1])
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = (rects[col], inputCentroids[col], outls[col])
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(rects[col], inputCentroids[col], outls[col])
        # return the set of trackable objects
        return self.objects
    
def fskel(mask, b_thresh=40, sk_thresh=20):
    '''
    Takes a mask in the form of a numpy boolean array. It returns a filfinder skeleton and a pruned filfinder skeleton.
    
    Parameters
    -------------
    mask = mask of an image in the form of a 2d numpy boolean array
    b_thresh = the branch threshold which will be used inputted into fil.analyze_skeletons
    sk_thresh = the skeletonization threshold which will be inputed into fil.analyze_skeletons
    '''
    mask = mask*255
    fil=FilFinder2D(mask,mask=mask)
    fil.preprocess_image(skip_flatten=True)
    fil.create_mask(use_existing_mask=True)
    fil.medskel(verbose=False)
    unpruned_skel = fil.skeleton
    fil.analyze_skeletons(branch_thresh=b_thresh*u.pix, skel_thresh=sk_thresh*u.pix, prune_criteria='length')
    skel = fil.skeleton_longpath
    length = fil.lengths().value[0]
    return unpruned_skel,skel, length

def intersection(line1,line2,width1=3,width2=3):
    '''
    Find if two lines intersect, or nearly intersect
    
    Parameteres
    -----------
    line1,line2=boolean arrays with 'True' values where the line exists
    width1,width2=amount (in pixels) the line should be dilated to see if a near intersection occurs
    '''
    m=np.shape(line1)[1]
    n=np.shape(line1)[0]
    d=m//2
    l_check = np.sum(line1[0:n,0:d]+line2[0:n,0:d]==2)>0
    r_check = np.sum(line1[0:n,d:m-1]+line2[0:n,d:m-1]==2)>0
    if l_check and r_check:
        return True
    else:
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(width1,width1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(width2,width2))
        dilated1 = cv2.dilate(line1.astype(np.uint8),kernel1,iterations=1)
        dilated2 = cv2.dilate(line2.astype(np.uint8),kernel2,iterations=1)
        m=np.shape(line1)[1]
        n=np.shape(line1)[0]
        d=m//2
        l_check = np.sum(dilated1[0:n,0:d]+dilated2[0:n,0:d]==2)>0
        r_check = np.sum(dilated1[0:n,d:m-1]+dilated2[0:n,d:m-1]==2)>0
        if l_check and r_check:
            return True
        else:
            return False

def skeleton_to_centerline(skeleton):
    '''
    Finds an ordered list of points which trace out a skeleton with no branches by following the pixels of the centerline 
    elementwise. Starts at the point farthest left and closest to the top. Ends when there are no new points on the skeleton 
    to go to.
    Parameters
    ----------
    skel = a 2d numpy boolean array representing a topological skeleton with no branches
    '''
    #initializing lists and starting parameters
    centerline=[]
    pos = bool_sort(skeleton)[0] # finds and sorts all points where the skeleton exists. Selects leftest as initial position
    centerline.append(pos) #appends initial position to the centerline
    skeleton[pos[0],pos[1]]=False #erases the previous positon
    local = make_local(skeleton,pos)
    while np.any(local): #checks if there are any new positions to go to
        pos = pos+bool_sort(local)[0]-np.array([1,1]) #updates position
        centerline.append(pos) #adds position to centerline
        skeleton[pos[0],pos[1]]=False #erases previous position
        local = make_local(skeleton,pos) #updates the local minor for the new position
    return centerline

def make_local(matrix,pos):
    '''
    Extracts the 3x3 array of values around a position in an existing array. If this is not possible (if the position 
    is along an edge), completes the array with rows/columns of 'False' values
    Paramters
    ---------
    matrix = a 2d numpy array
    pos = a numpy array or list, indicating a position in array
    '''
    N = np.shape(matrix)[0]+1
    M = np.shape(matrix)[1]+1
    n1 = max(0,pos[0]-1)
    n2 = min(N,pos[0]+2)
    m1 = max(0,pos[1]-1)
    m2 = min(M,pos[1]+2)
    if n1 == pos[0]-1 and n2 != N and m1==pos[1]-1 and m2!=M:
        return matrix[n1:n2,m1:m2]
    elif n1 == 0:
        if m1 == 0:
            return np.hstack((np.array([[False],[False],[False]]),np.vstack((np.array([False,False]), matrix[n1:n2,m1:m2]))))
        elif m2 == M:
            return np.hstack((np.vstack((np.array([False,False]), matrix[n1:n2,m1:m2])),np.array([[False],[False],[False]])))
        else:
            return np.vstack((np.array([False,False,False]), matrix[n1:n2,m1:m2]))
    elif n2 == N:
        if m1 == 0:
            return np.hstack((np.array([[False],[False],[False]]),np.vstack((matrix[n1:n2,m1:m2],np.array([False,False])))))
        elif m2==M:
            return np.hstack((np.vstack((matrix[n1:n2,m1:m2],np.array([False,False]))),np.array([[False],[False],[False]])))
        else:
            return np.vstack((matrix[n1:n2,m1:m2],np.array([False,False,False])))
    elif m1==0:
        return np.hstack((np.array([[False],[False],[False]]),matrix[n1:n2,m1:m2]))
    elif m2 ==M:
        return np.hstack((matrix[n1:n2,m1:m2],np.array([[False],[False],[False]])))

def bool_sort(array):
    '''
    Sort the "True" values in a boolean array, starting from left to right, then top to bottom
    Parameters
    ---------
    array = a boolean array to be sorted
    '''
    output = np.transpose(np.array(list(np.where(array))))
    output = output[output[:,0].argsort()]
    output = output[output[:,1].argsort(kind='mergesort')]
    return output