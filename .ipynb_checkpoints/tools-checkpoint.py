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
import sknw
from networkx import shortest_path
from scipy.signal import argrelmax
from scipy.interpolate import splprep
from scipy.interpolate import splev

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

def get_overlap(outl, boxes):
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
    return cell_overlaps 

def peri_area(outl):
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
    peris = []
    areas = []
    for o in outl:
        # Calculate the perimeter and area for each outline and 
        # append values to lists. 
        peris.append(cv2.arcLength(o, True))
        areas.append(cv2.contourArea(o))
    return peris, areas

def save_ind_masks(path, IDs_list, outl_new_list, time_list, img_list):
    '''
    A function which takes a set of images with associated timepoints, cell outlines, and cell IDs 
    and then isolates individual cell images and masks to be saved at a specified path. It is 
    designed to automatically organize the data within the target directory so that each image
    has a folder named after the time the image was taken and contains seperate subfolders for the 
    individual cell images, masks, and skeletons. 

    Parameters
    ---------------
    path: string
    A string which specifies the path we want to save our individual cell images and masks. 
    
    IDs_list: list
    A nested list that contains a list of cell IDs for each image. 
    
    outl_new_list: list
    A nested list that contains a list of outlines for each cell in each image. 
    
    time_list: list
    A list of timepoints for each image. 
    
    img_list: list 
    a list of images. 
    '''
    # Iterate over every image and its associated time point, set of cell ids, and set of cell outlines.
    for ID_set, outl, time, img in zip(IDs_list, outl_new_list, time_list, img_list):
        # For each image and timepoint, make sure that there is a location to save the 
        # cell outline, mask, and skeleton. 
        if os.path.exists(path + str(time)) == False:
            os.makedirs(path + str(time))
        if os.path.exists(path + str(time) + "/cells") == False:
            os.makedirs(path + str(time) + "/cells")
        if os.path.exists(path + str(time) + "/masks") == False:
            os.makedirs(path + str(time) + "/masks")
        if os.path.exists(path + str(time) + "/skeletons") == False:
            os.makedirs(path + str(time) + "/skeletons")
        # Iterate over each cell in each image. 
        for idx, cell in zip(ID_set, outl):
            # mask outline
            mask = np.zeros(img.shape, dtype=np.uint8)
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(mask, [cell], ignore_mask_color)
            masked_image = cv2.bitwise_and(img, mask)
    
            # crop the cell
            x = cell.flatten()[::2]
            y = cell.flatten()[1::2]
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            out_cell = masked_image[topy:bottomy+1, topx:bottomx+1]
            out_mask = mask[topy:bottomy+1, topx:bottomx+1]
            
            # save the individual cell image and cell mask. 
            im_mask = Image.fromarray(out_mask)
            im_mask.save(path + str(time) + "/masks" + "/" + "mask_" + str(idx) + ".png")
            cv2.imwrite(path + str(time) + "/cells" + "/" + "cell_" + str(idx) + ".png", out_cell)
            
def skeleton_length(path, ID_set, time):
    '''
    This iterative function loads up a set of skeletons for individual cells which were created 
    from a set of parent images taken at different timepoints. For each of these skeletons, it uses
    the filfinder package to convert the skeleton to a filfinder skeleton. This fairly complicated
    process is done simply so that the length of the skeleton can be recorded. 
    
    Parameters
    ---------------
    path: string
    A string which contains the path to the individual cell images and cell masks. 
    
    ID_set: list
    list that contains the cell IDs for an image.
    
    time: integer
    The time point for an image. 
    
    Returns
    ---------------
    length_set: list
    A list which contains the length of each cell from an image as a float with pixel units. 
    '''
    # Initialize a list to hold skeleton lengths for each individual cell.
    length_set = []
    for idx in (ID_set):
        # Load the skeleton as a list of coordinates and then convert to an image array. 
        skeleton = np.loadtxt(open(path + str(time) + "/skeletons/matrix_" + str(idx) + ".csv", "rb"), delimiter=",", skiprows=1)[1:-2].astype(int)
        dbl = int((len(skeleton)-2)/2)
        cut_skeleton = skeleton[1:-2]

        x, y = [i[0] for i in cut_skeleton], [i[1] for i in cut_skeleton]

        fil_mask=imageio.imread(path + str(time) + "/masks/mask_" + str(idx) + ".png")
        fil_skeleton = np.zeros(fil_mask.shape, dtype = 'uint8')

        for i in range(len(cut_skeleton)):
            fil_skeleton[y[i], x[i]] = 1
            
        # Convert the fil_skeleton to grayscale.
        fil_skeleton = cv2.cvtColor(fil_skeleton, cv2.COLOR_BGR2GRAY)
            
        # Convert the original skeleton into a filfinder spine. 
        fil = FilFinder2D(fil_skeleton, distance=250 * u.pix, mask=fil_skeleton)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False,
        use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
        
        # Extract the length of the skeleton. 
        length = fil.lengths().value[0]
        length_set.append(length)
    return(length_set)        
    
def radobj_maker(path, ID_set, time):
    '''
    This iterative function loads up a set of images, masks and skeletons for individual cells which were created 
    from a set of parent images taken at different timepoints. For each of these individual cells, it then uses
    the RadFil package to estimate the radial profile of the cell which is then saved in a list of RadFil objects.
    
    Parameters
    ---------------
    path: string
    A string which contains the path to the individual cell images and cell masks. 
    
    ID_set: list
    list that contains the cell IDs for an image.
    
    time: integer
    The time point for an image. 
    
    Returns
    ---------------
    radobj_set: list
    A list which contains a RadFil object for each cell in an image. 
    '''

    radobj_set = []
    for idx in (ID_set):
        # Load the image and convert to grayscale. 
        fil_image =imageio.imread(path + str(time) + "/cells/cell_" + str(idx) + ".png")
        fil_image = cv2.cvtColor(fil_image, cv2.COLOR_BGR2GRAY)
            
        # Load the mask, convert to grayscale, and then convert the mask into a boolean array.
        fil_mask=imageio.imread(path + str(time) + "/masks/mask_" + str(idx) + ".png")
        fil_mask = cv2.cvtColor(fil_mask, cv2.COLOR_BGR2GRAY)
        fil_mask = np.array(fil_mask, dtype=bool)

        # Load the skeleton as a list of coordinates and then convert to a boolean array. 
        skeleton = np.loadtxt(open(path + str(time) + "/skeletons/matrix_" + str(idx) + ".csv", "rb"), delimiter=",", skiprows=1)[1:-2].astype(int)
        dbl = int((len(skeleton)-2)/2)
        cut_skeleton = skeleton[1:-2]

        x, y = [i[0] for i in cut_skeleton], [i[1] for i in cut_skeleton]

        fil_skeleton = np.zeros(fil_mask.shape, dtype = bool)

        for i in range(len(cut_skeleton)):
            fil_skeleton[y[i], x[i]] = True
            
        # Use RadFil to create a radial profil object and then append this object to a list.
        # The data stored in this object will be used to calculate height and radial profile along the
        # medial axis. 
        radobj=radfil_class.radfil(fil_image, mask=fil_mask, filspine=fil_skeleton, distance=200)
        radobj.build_profile(samp_int=1, shift = False)
        radobj_set.append(radobj)
    return(radobj_set)

def find_nearest(array, value):
    '''
    Finds the nearest value in an array and returns the index of that value.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def radobj_extractor(radobj_list, h_conv = 3.9215686274509802):
    '''
    This function iterates through a list of RadFil objects and extracts two meaningful outputs for each cell.
    The first is a list that details the diameter of the cell along the medial axis, and the second is a list
    that details the pixel intensity of the original image along the medial axis. This second list will reflect
    whatever the original image was measuring (e.g. height, stiffness, brightness). 
    
    Parameters
    ---------------
    radobj_list: nested list
    A nested list with a structure of: 
    
    Returns
    ---------------
    diam_list: list
    a nested list which contains lists of radial profiles for each cell in each image. 
    
    height_list: list
    A nested list which contains lists of ridgeline height for each cell in each image. 
    '''
    # Initialize lists to hold the data for each image.
    diam_list = []
    height_list = []
    profile_list = []
    dist_list = []
    for radobj_set in radobj_list:
        # Initialize lists to hold the data for each individual cell
        diam_set = []
        height_set = []
        profile_set = []
        dist_set = []
        for radobj in radobj_set:
            # Initialize lists to hold the data for each point along the skeleton. 
            diam = []
            height = []
            profile = []
            distance = []
            for dist, prof in zip(radobj.dictionary_cuts['distance'],radobj.dictionary_cuts['profile']) :
                # Calculate the diameter along the skeleton.
                diam.append(np.abs(dist[0] - dist[-1]))
                
                # Calculate the pixel intensity along the skeleton and convert to height. 
                ind = find_nearest(dist,0)
                height.append(prof[ind]*h_conv)
                profile.append(prof*h_conv)
                distance.append(dist)
            
            profile_set.append(profile)
            diam_set.append(diam)
            height_set.append(height)
            dist_set.append(distance)
        profile_list.append(profile_set)
        diam_list.append(diam_set)
        height_list.append(height_set)
        dist_list.append(dist_set)
    return diam_list, height_list, profile_list, dist_list

def get_max_ID(IDs_list):
    '''
    A function which finds the final cell ID number. 
    '''
    maxi = []
    for ID_set in IDs_list:
        maxi.append(max(ID_set))
    return max(maxi)

def get_metadata(exact_ID, IDs_list, per_list, area_list,
                 overl_list, centers_list, time_list,
                diam_list, height_list, profile_list,
                 dist_list, length_list):
    '''
    Parameters
    ---------------  
    exact_ID: integer
    The function will generate a metadata table for a cell with this ID. 
    
    IDs_list: list
    A nested list which contains a list of cell IDs for each original input image. 
    
    per_list: list
    A nested list which contains the perimeter size of a cell in pixels for each cell in each original input image.
    
    area_list: list
    A nested list which contains the area of a cell in pixels for each cell in each original input image.
    
    overl_list: list
     A nested list which contains the degree of overlap for each cell in each original input image.
    
    centers_list: list
    A nested list which contains lists of centroid coordinates for each cell in each original input image. 
    
    time_list: list
    A list of timepoints for each image. 
    
    diam_list: list
    A nested list which contains lists of radial profiles for each cell in each original input image. 
    
    height_list: list
    A nested list which contains lists of ridgeline height for each cell in each original input image. 
    
    length_list: list
    A nested list which contains the length of each cell skeleton in each original input image. 
    
    Returns
    ---------------
    df: pandas dataframe
    A dataframe with parameters as columns (perimeter, area, etc) and timepoints as rows. 
    '''
    data = []
    for ID_set, per_set, area_set, overl_set, centers_set, time, diam_set, height_set, profile_set, dist_set, length_set in zip(IDs_list,
                                    per_list, area_list, overl_list, centers_list, time_list,
                                    diam_list, height_list, profile_list, dist_list, length_list):
        if exact_ID in ID_set:
            for idx, per, area, overl, center, diam, height, profile, dist, length in zip(ID_set,
                                            per_set, area_set, overl_set, centers_set,
                                            diam_set, height_set, profile_set, dist_set, length_set):
                if idx == exact_ID:
                    data.append(dict(zip(["time", "perimeter", "area", "overl", "location",
                                     "diameter profile (pixels)", "Height (nm)", "Height Profile (nm)",
                                          "Distance Profile (nm)", "length (pixels)"],
                                        [time, per, area, overl, center, diam, height, profile, dist, length])))
        else:
            data.append(dict(zip(["time", "perimeter", "area", "overl", "location",
                                     "diameter profile (pixels)", "Height (nm)",
                                  "Height Profile (nm)", "Distance Profile (nm)", "length (pixels)"], 
                                 [time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])))
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset = "time", ignore_index = True)
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
    mask = mask of an image (opencv image)
    b_thresh = the branch threshold which will be used inputted into fil.analyze_skeletons
    sk_thresh = the skeletonization threshold which will be inputed into fil.analyze_skeletons
    '''
    mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)[:,:,0]>0 #add a border to the skel
    fil=FilFinder2D(mask,mask=mask)
    fil.preprocess_image(skip_flatten=True)
    fil.create_mask(use_existing_mask=True)
    fil.medskel(verbose=False)
    unpruned_skel = fil.skeleton
    unpruned_skel = unpruned_skel[20:np.shape(unpruned_skel)[0]-20,20:np.shape(unpruned_skel)[1]-20]
    fil.analyze_skeletons(branch_thresh=b_thresh*u.pix, skel_thresh=sk_thresh*u.pix, prune_criteria='length')
    skel = fil.skeleton_longpath
    skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20] #remove borders from skeletons
    return unpruned_skel,skel

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

def skeleton_to_centerline(skeleton,top=False):
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
    pos0 = bool_sort(skeleton,top)[0] # finds and sorts all points where the skeleton exists. Selects leftest as initial position
    pos = np.array(pos0)
    centerline.append(pos) #appends initial position to the centerline
    skeleton[pos[0],pos[1]]=False #erases the previous positon
    local = make_local(skeleton,pos)
    while np.any(local): #checks if there are any new positions to go to
        pos = pos+bool_sort(local)[0]-np.array([1,1]) #updates position
        centerline.append(pos) #adds position to centerline
        skeleton[pos[0],pos[1]]=False #erases previous position
        local = make_local(skeleton,pos) #updates the local minor for the new position
    
    local = make_local(skeleton,pos0)
    if np.any(local): #check if another part of the centerline emerges from pos0
        centerline2=[] #initialize a new section of the centerline
        pos = np.array(pos0) #update position and restart the process
        while np.any(local): #checks if there are any new positions to go to
            pos = pos+bool_sort(local)[0]-np.array([1,1]) #updates position
            centerline2.append(pos) #adds position to centerline
            skeleton[pos[0],pos[1]]=False #erases previous position
            local = make_local(skeleton,pos) #updates the local minor for the new position
        
        if centerline[-1][0]<centerline2[-1][0]:
            centerline.reverse()
            return centerline + centerline2
        else:
            centerline2.reverse()
            return centerline2+centerline
    else:
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

def bool_sort(array,top=False):
    '''
    Sort the "True" values in a boolean array, starting from left to right, then top to bottom. If top is true, then start from top to bottom, then
    then left to right.
    Parameters
    ---------
    array = a boolean array to be sorted
    top = whether algorithm starts at top of the figure, or the left. Default is False.
    '''
    if top:
        array = np.transpose(array)
        output = list(np.where(array))
        output.reverse()
        output = np.transpose(np.array(output))
        output = output[output[:,1].argsort()]
        output = output[output[:,0].argsort(kind='mergesort')]
    else:
        output = np.transpose(np.array(list(np.where(array))))
        output = output[output[:,0].argsort()]
        output = output[output[:,1].argsort(kind='mergesort')]
    return output

def pts_to_img(pts,base):
    '''converts a list of points to a binary image, with the dimensions of the skeleton'''
    path_in_img=np.zeros(np.shape(base))>0
    for pt in pts:
        path_in_img[pt[0],pt[1]]=True
    return path_in_img        

def prune2(skel,outline,poles,sensitivity=20,crop=0.1):
    def intersection(line1,line2,width1=3,width2=3):
        '''
        Find if two lines intersect, or nearly intersect

        Parameteres
        -----------
        line1,line2=boolean arrays with 'True' values where the line exists
        width1,width2=amount (in pixels) the line should be dilated to see if a near intersection occurs
        '''
        if np.sum(line1+line2==2)>0:
            return True
        else:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(width1,width1))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(width2,width2))
            dilated1 = cv2.dilate(line1.astype(np.uint8),kernel1,iterations=1)
            dilated2 = cv2.dilate(line2.astype(np.uint8),kernel2,iterations=1)
            return np.sum(dilated1+dilated2==2)>0
    def crop_centerline(centerline):
        '''crops centerline to desired length'''
        image = centerline.copy()
        points = skeleton_to_centerline(image,not(long))
        crop_length=round(len(points)*crop)
        if long:
            if true_starts==[]:
                left_centerline=centerline[:,0:n//2]
                left_outline=outline[:,0:n//2]
                if intersection(left_centerline,left_outline):
                    points= points[crop_length:]
            if true_ends==[]:      
                right_centerline = centerline[:,n//2:n]
                right_outline = outline[:,n//2:n]
                if intersection(right_centerline,right_outline): 
                    points = points[:len(points)-crop_length]
            return pts_to_img(points,skel)
        else:
            if true_starts==[]:
                upper_centerline=centerline[0:m//2,:]
                upper_outline=outline[0:m//2,:]
                if intersection(upper_centerline,upper_outline):
                    points= points[crop_length:]
            if true_ends==[]:      
                lower_centerline = centerline[m//2:m,:]
                lower_outline = outline[m//2:m,:]
                if intersection(lower_centerline,lower_outline): 
                    points = points[:len(centerline)-crop_length]
                return pts_to_img(points,skel)
    def find_splines(centerline):
        image = centerline.copy()
        points = skeleton_to_centerline(image,not(long))
        if true_starts == []:
            if true_ends == []:
                points = [start_pole] + points + [end_pole]
                tck,u=splprep(np.transpose(points),ub=start_pole,ue=end_pole)
                [ys,xs]=splev(u,tck)
            else:
                points = [start_pole] + points
                tck,u=splprep(np.transpose(points),ub=start_pole)
                [ys,xs]=splev(u,tck)
        elif true_ends ==[]:
            points = points + [end_pole]
            tck,u=splprep(np.transpose(points),ub=start_pole,ue=end_pole)
            [ys,xs]=splev(u,tck)
        else:
            tck,u=splprep(np.transpose(points))
            [ys,xs]=splev(u,tck)
        return [xs,ys],u
    if np.any(np.isnan(np.array(poles))):
        raise ValueError('Poles must be numerical values, not np.NaN')
    #initializing parameters
    m = np.shape(skel)[0]
    n = np.shape(skel)[1]
    long = m<n #True if long axis is horizontal
    graph = sknw.build_sknw(skel) #creating the graph from the skeleton
    all_paths=shortest_path(graph) #shortest possible paths between the nodes in the skeleton
    nodes = graph.nodes()
    paths = []
    centerlines=[]
    curvatures=[]
    
    #initializing suitable starting and ending positions for the skeleton
    if long:
        poles.sort(key = lambda pole: pole[0])
        start_pole = np.array([poles[0][1],poles[0][0]])
        end_pole = np.array([poles[1][1],poles[1][0]])
        start_nodes = list([i for i in nodes if nodes[i]['o'][1]<n//2])
        end_nodes= list([i for i in nodes if nodes[i]['o'][1]>n//2])
    else:
        poles.sort(key = lambda pole: pole[1])
        start_pole = np.array([poles[0][1],poles[0][0]])
        end_pole = np.array([poles[1][1],poles[1][0]])
        start_nodes = list([i for i in nodes if nodes[i]['o'][0]<m//2])
        end_nodes= list([i for i in nodes if nodes[i]['o'][0]>m//2])
    
    # if there are some nodes close to the poles, check only those nodes
    true_starts = [i for i in start_nodes if np.linalg.norm(nodes[i]['o']-np.array([poles[1][1],poles[1][0]]))<sensitivity]
    true_ends = [i for i in end_nodes if np.linalg.norm(nodes[i]['o']-np.array([poles[0][1],poles[0][0]]))<sensitivity]
    if true_starts != []:
        start_nodes = true_starts
    if true_ends !=[]:
        end_nodes = true_ends
    if start_nodes == [] or end_nodes == []:
        raise ValueError('skeleton is on only one side of the cell')
    # take all paths between starting and ending nodes
    for b in  start_nodes:
        for e in end_nodes:
            path = all_paths[b][e]
            paths.append(path)
    if len(paths) == 1:
        path = paths[0]
        edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
        centerline_path = []
        for (b,e) in edges:
            edge = graph[b][e]['pts']
            centerline_path = centerline_path + list(edge)
        centerline = pts_to_img(centerline_path,skel)
        if not (np.any(centerline)):
            raise ValueError('Skeleton has been erased')
        centerline = crop_centerline(centerline)
        if not (np.any(centerline)):
            raise ValueError('Skeleton has been erased')
        return centerline,find_splines(centerline)
    #convert paths (lists of nodes) to centerlines (lists of points)
    for path in paths:
        edges = [(path[i],path[i+1]) for i in range(len(path)-1)] #edges of the graph corresponding to the centerline
        #initializing centerline
        centerline_path = []
        #calling points from the graph
        for (b,e) in edges:
            edge = graph[b][e]['pts']
            centerline_path = centerline_path + list(edge)
        #convert path to binary image
        centerline=pts_to_img(centerline_path,skel)
        if not (np.any(centerline)):
            raise ValueError('Skeleton has been erased')
        #crop the centerline, if it has a false pole
        centerline = crop_centerline(centerline)
        if not (np.any(centerline)):
            raise ValueError('Skeleton has been erased')
        # add to the list of centerlines
        centerlines.append(centerline)
    #calculate the maximum curvatures of each possible centerline
    for centerline in centerlines:
        centerline_image = centerline.copy()
        centerline_pts = skeleton_to_centerline(centerline_image,not(long))
        if not (np.any(centerline)):
            raise ValueError('Skeleton has been erased')
        tck,u = splprep(np.transpose(centerline_pts))
        V=np.array(splev(u,tck,der=1)).T
        A=np.array(splev(u,tck,der=2)).T
        K = [abs(np.cross(V[i],A[i]))/(np.linalg.norm(V[i])**3) for i in range(0,len(u))]
        curvatures.append(max(K))
    #choose the centerline with the least maximum curvature
    min_index=curvatures.index(min(curvatures))
    centerline=centerlines[min_index]
    if not (np.any(centerline)):
        raise ValueError('Skeleton has been erased')
    return centerline, find_splines(centerline)

def radii(x,y):
    '''
    Finds the radius function of a 2d curve
    '''
    length = len(y)
    centroid = np.array([np.sum(x),np.sum(y)])/length
    vectors = np.transpose(np.array([x,y]))-centroid
    return np.array([np.linalg.norm(v) for v in vectors]),centroid

def explore_poles(x,y,long=True):
    '''
    Finds the poles (average distance of the farthest points from the centroid on a smooth closed curve) from x and y coords
    Parameters
    ----------
    x = x coordinates of the curve
    y = y coordinates of the curve
    long = whether the cell is oriented lengthwise (default True)
    '''
    r,centroid = radii(x,y)
    cx = centroid[0]
    cy = centroid[1]
    peaks = argrelmax(r)[0]
    if len(peaks)<2:
        peaks = argrelmax(r,mode='wrap')[0]
    if long:
        right_x_pos=[x[i] for i in peaks if x[i]>cx]
        right_y_pos=[y[i] for i in peaks if x[i]>cx]
        right_rads=[r[i] for i in peaks if x[i]>cx]
        left_x_pos=[x[i] for i in peaks if x[i]<cx]
        left_y_pos=[y[i] for i in peaks if x[i]<cx]
        left_rads=[r[i] for i in peaks if x[i]<cx]
        
        average_x = np.array([np.dot(right_x_pos,right_rads)/sum(right_rads), np.dot(left_x_pos,left_rads)/sum(left_rads)])
        average_y = np.array([np.dot(right_y_pos,right_rads)/sum(right_rads), np.dot(left_y_pos,left_rads)/sum(left_rads)])
        return np.transpose([average_x,average_y]), centroid
    else:
        lower_x_pos=[x[i] for i in peaks if y[i]>cy]
        lower_y_pos=[y[i] for i in peaks if y[i]>cy]
        lower_rads=[r[i] for i in peaks if y[i]>cy]
        upper_x_pos=[x[i] for i in peaks if y[i]<cy]
        upper_y_pos=[y[i] for i in peaks if y[i]<cy]
        upper_rads=[r[i] for i in peaks if y[i]<cy]
        
        average_x = np.array([np.dot(lower_x_pos,lower_rads)/sum(lower_rads), np.dot(upper_x_pos,upper_rads)/sum(upper_rads)])
        average_y= np.array([np.dot(lower_y_pos,lower_rads)/sum(lower_rads), np.dot(upper_y_pos,upper_rads)/sum(upper_rads)])
        return np.transpose([average_x,average_y]), centroid