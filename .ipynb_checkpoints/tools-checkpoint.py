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
import imageio.v2 as imageio
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
from cellpose import utils
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from skimage.morphology import skeletonize
import warnings

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
        fil_skeleton = imageio.imread(path + str(time) + '/skeletons/skeleton_' + str(idx) + '.png')>0
        #skeleton = np.loadtxt(open(path + str(time) + "/skeletons/matrix_" + str(idx) + ".csv", "rb"), delimiter=",", skiprows=1)[1:-2].astype(int)
        #dbl = int((len(skeleton)-2)/2)
        #cut_skeleton = skeleton[1:-2]

        #x, y = [i[0] for i in cut_skeleton], [i[1] for i in cut_skeleton]

        #fil_skeleton = np.zeros(fil_mask.shape, dtype = bool)

        #for i in range(len(cut_skeleton)):
        #    fil_skeleton[y[i], x[i]] = True
           
        # Use RadFil to create a radial profil object and then append this object to a list.
        # The data stored in this object will be used to calculate height and radial profile along the
        # medial axis. 
        radobj=radfil_class.radfil(fil_image, mask=fil_mask, filspine=fil_skeleton, distance=200)
        radobj.build_profile(samp_int=1, shift = False)
        radobj_set.append(radobj)
    return(radobj_set)


def extract_ind_cells(IDs_list, outl_list, height_img_list):
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
    #ind_cell_stiff_list = []
    #ind_cell_pfe_list = [] 
    ind_cell_mask_list = []
    for ID_set, outl, h_img in zip(IDs_list, outl_list, height_img_list):
        ind_cell_height = []
        #ind_cell_stiff = []
        #ind_cell_pfe = [] 
        ind_cell_mask = []
        # Iterate over each cell in each image. 
        for idx, cell in zip(ID_set, outl):
            # extract the mask from the
            mask = np.zeros(h_img.shape, dtype=np.uint8)
            if len(h_img.shape)>2:
                channel_count = h_img.shape[2]  # i.e. 3 or 4 depending on your image
            else:
                channel_count = 1
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
            #if np.isnan(s_img.any()) == False:
            #    out_stiff = cv2.warpPerspective(s_img, M, (width, height)) 
            #    out_stiff = cv2.copyMakeBorder(out_stiff, top = 5, bottom = 5, left = 5, right = 5,
            #                                    borderType=cv2.BORDER_CONSTANT, value = [0])
            #else:
            #    out_stiff = np.nan
                
            #if np.isnan(p_img.any()) == False:
            #    out_pfe = cv2.warpPerspective(p_img, M, (width, height))
            #    out_stiff = cv2.copyMakeBorder(out_stiff, top = 5, bottom = 5, left = 5, right = 5,
            #                                   borderType=cv2.BORDER_CONSTANT, value = [0])
            #else:
                #out_pfe = np.nan 
                
            # If the height is greater than the width, rotate it so that the long side is horizontal. 
            if height > width:
                out_height = cv2.transpose(out_height)
                out_mask = cv2.transpose(out_mask) 
                #out_stiff = cv2.transpose(out_stiff) 
                #out_pfe = cv2.transpose(out_pfe) 
            
            # Append everything into new nested lists. 
            ind_cell_height.append(out_height)
            #ind_cell_stiff.append(out_stiff)
            #ind_cell_pfe.append(out_pfe)
            ind_cell_mask.append(out_mask)
        ind_cell_height_list.append(ind_cell_height)
        #ind_cell_stiff_list.append(ind_cell_stiff)
        #ind_cell_pfe_list.append(ind_cell_pfe)
        ind_cell_mask_list.append(ind_cell_mask)
    return(ind_cell_height_list, ind_cell_mask_list)

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
            if len(image.shape)>2:
                fil_image = image[:,:,0]
                fil_mask = mask[:,:,0]>0
            else:
                fil_image = image
                fil_mask = mask>0
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
    Takes a mask in the form of a numpy boolean array. It returns a filfinder skeleton.
    
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
    return unpruned_skel

def padskel(mask):
    '''
    Runs skimage.morphology.skeletonize on a padded version of the mask, to avoid errors.
    
    Parameters
    ----------
    mask = a mask (opencv image)
    
    Returns
    -------
    skel = a skeleton (boolean array)
    '''
    if len(mask.shape)>2:
        mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)[:,:,0]>0 #add a border to the skel
    else:
        mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)>0 #add a border to the skel
    skel = skeletonize(mask)
    skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20]
    return skel

def intersection(line1,line2,width1=3,width2=3):
    '''
    Find if two lines intersect, or nearly intersect
    
    Parameteres
    -----------
    line1,line2=boolean arrays with 'True' values where the line exists
    width1,width2=amount (in pixels) the line should be dilated to see if a near intersection occurs
    '''
    check = np.sum(line1+line2==2)>0
    if check:
        return True
    else:
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(width1,width1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(width2,width2))
        dilated1 = cv2.dilate(line1.astype(np.uint8),kernel1,iterations=1)
        dilated2 = cv2.dilate(line2.astype(np.uint8),kernel2,iterations=1)
        check = np.sum(dilated1+dilated2==2)>0
        return check

def skeleton_to_centerline(skeleton,axis=False):
    '''
    Finds an ordered list of points which trace out a skeleton with no branches by following the pixels of the centerline 
    elementwise. Starts at the point farthest left and closest to the top. Ends when there are no new points on the skeleton 
    to go to.
    Parameters
    ----------
    skel = a 2d numpy boolean array representing a topological skeleton with no branches
    axis = if true start by finding the min point on the 0 (top-bottom) axis, then 0 (left-right) axis. If false, vice verse. Default False.
    '''
    #initializing lists and starting parameters
    centerline = bool_sort(skeleton,axis) # finds and sorts all points where the skeleton exists. Selects leftest as initial position
    centerline = reorder_opt(centerline)
    return list(centerline)

def reorder_opt(points,NN=5):
    '''
    Find the optimal order of a set of points in an image
    
    Paramters
    ---------
    points = an Nx2 array-like corresponing to a set of points
    
    Returns
    --------
    points = an ordered list of points
    '''
    #find the nearest neighbour of each point
    points = np.array(points)
    clf=NearestNeighbors(n_neighbors=NN).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_array(G)

    mindist = np.inf
    minidx = 0

    order = list(nx.dfs_preorder_nodes(T, 0))
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
    
    for i in range(len(points)):
        p = paths[i]           # order of nodes
        ordered = points[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i
    opt_order = paths[minidx]
    return list(points[opt_order,:])

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

def bool_sort(array,axis=True):
    '''
    Sort the "True" values in a boolean array, starting from left to right, then top to bottom. If top is true, then start from top to bottom, then
    then left to right.
    Parameters
    ---------
    array = a boolean array to be sorted
    axis = which axis is whose lowest values are put first, default True. If True put the 0 (top-bottom) axis first, if False put the 1 (left-right) axis first.
    '''
    if axis:
        output = np.transpose(np.where(array))
        output = output[output[:,1].argsort()]
        output = output[output[:,0].argsort(kind='mergesort')]
    else:
        output = np.transpose(np.where(array))
        output = output[output[:,0].argsort()]
        output = output[output[:,1].argsort(kind='mergesort')]
    return output

def pts_to_img(pts,base):
    '''converts a list of points to a binary image, with the dimensions of the skeleton'''
    path_in_img=np.zeros(np.shape(base)).astype(bool)
    for pt in pts:
        pt = pt.astype(np.uint16)
        path_in_img[pt[0],pt[1]]=True
    return path_in_img 

def order_by_NN(points,source=None,destination=None,thresh=5):
    ordered = []
    if source is None:
        source = points[0]
    ordered.append(source)
    pos = source

    while points != []:
        next_pos,r = find_NN(pos,points)
        if r>thresh:
            warnings.warn('Sorting terminated early when points exceeded allowable distance')
            break
        ind =np.where(np.array([np.all(p == next_pos) for p in points]))
        points = list(np.delete(np.array(points),ind,axis=0))
        ordered.append(next_pos)
        pos = next_pos
    if np.linalg.norm(ordered[-1]-destination)>2 and not destination is None:
        warnings.warn('path did not reach destination')
    return ordered

def find_NN(point,posl):
    NN = posl[0]
    rmax = np.linalg.norm(posl[0]-point)
    for pos in posl:
        r = np.linalg.norm(pos-point)
        if r<rmax:
            NN = pos
            rmax = r
    return NN,rmax

def prune2(skel,outline,mask,poles,sensitivity=5,crop=0.1,k_thresh=0.2):
    '''Creates a centerline of the cell from the skeleton, removing branches and extending the centerline to the poles
    skel = topological skeleton of the cell (boolean array)
    outline = outline of the cell (boolean array)
    mask = mask of the cell (booelan array)
    poles = two poles of the cell. Ordered as [desired start pole of the centerline, desired end pole of the centerline]
    sensitivity = distance to a pole (in pixels) which a branch of the skeleton must be to be considered to "reach" that pole. Default is 5 pixels
    crop = proportion of the centerline to crop at a false pole (an end of a branch which intersects the outline not at a pole). Default is 0.1
    k_thresh = curvature (radius of curvature^{-1}) threshold for the final spline. above this the centerline will be cropped for a second time to ensure
    the ends of the centerline don't have abnormally high curvature. Default = 0.2 (corresponding to a radius of curvature of 5 pixels)
    '''
    def crop_centerline(centerline_path):
        '''Helper function. Crops a proportion of the points (determined by the crop parameter) from the ends of a centerline with a false pole
        Paramters:
        --------
        centerline_path = list of points (numpy arrays) on a line in order of the arclength parametrization, initiated at the start node
        '''
        crop_length=round(len(centerline_path)*crop)
        image = pts_to_img(centerline_path,skel)
        if true_starts == []:
            if axis:
                start_centerline = image[:,0:n//2]
                start_outline = outline[:,0:n//2]
            else:
                start_centerline=image[0:m//2,:]
                start_outline=outline[0:m//2,:]
            if intersection(start_centerline,start_outline):
                #print('Starts',crop_length)
                centerline_path= centerline_path[crop_length:]

        if true_ends == []:
            if axis:
                end_centerline = image[:,n//2:n]
                end_outline = outline[:,n//2:n]
            else:
                end_centerline = image[m//2:m,:]
                end_outline = outline[m//2:m,:]
            if intersection(end_centerline,end_outline): 
                #print('Ends',crop_length)
                centerline_path = centerline_path[:len(centerline_path)-crop_length]
        return centerline_path
    def find_splines(points):
        '''
        Helper function. Return spline of a centerline based on a set of ordered points.
        Parameters
        --------
        points = list of points (numpy arrays) on a line in order of the arclength parametrization, initiated at the start node
        '''
        s = np.linspace(0,1,1000)
        # remove repeated points
        diff = np.diff(points,axis=0)==0
        repeated = np.where(np.array([pt[0] and pt[1] for pt in diff]))
        points = list(np.delete(points,repeated,axis=0))
        if true_starts ==[]:
            points = [np.array([start_pole[1],start_pole[0]])] + points
        if true_ends ==[]:
            points = points + [np.array([end_pole[1],end_pole[0]])]
        tck,U=splprep(np.transpose(points))
        s = np.linspace(0,1,1000)
        [ys,xs]= splev(s,tck)
        #calculate the curvature of the spline
        v = np.transpose(splev(s,tck,der=1))
        a = np.transpose(splev(s,tck,der=2))
        k=abs(np.cross(v,a)/np.array([np.linalg.norm(V)**3 for V in v]))
        # check if the curvature exceeds a threshold
        if np.any(k>k_thresh):
            warnings.warn('curvature threshold exceeded. additional pruning executed in response')
            crop_length=round(len(points)*crop)
            if true_starts == []:
                points= points[crop_length:]
            if true_ends == []:
                points = points[:len(points)-crop_length]
            #create a new spline
            if true_starts ==[]:
                points = [np.array([start_pole[1],start_pole[0]])] + points
            if true_ends ==[]:
                points = points + [np.array([end_pole[1],end_pole[0]])]
            #remove repeated points
            diff = np.diff(points,axis=0)==0
            repeated = np.where(np.array([pt[0] and pt[1] for pt in diff]))
            points = list(np.delete(points,repeated,axis=0))
            #create a new spline
            tck,U=splprep(np.transpose(points))
            s = np.linspace(0,1,1000)
            [ys,xs]= splev(s,tck)
            v = np.transpose(splev(s,tck,der=1))
            a = np.transpose(splev(s,tck,der=2))
            k=abs(np.cross(v,a)/np.array([np.linalg.norm(V)**3 for V in v]))
            if np.any(k>k_thresh):
                warnings.warn('Curvature exceeds threshold')
        return [xs,ys],s
    def in_mask (path,mask):
        '''
        Returns values of a path which are in a mask
        Parameters:
        ---------
        path = numpy array consisting of coordinates in the mask
        mask = numpy boolean array, showing the mask
        '''
        #first remove all values that are not in the picture
        not_in_img =[not(pt[0]<np.shape(mask)[0]-1 and pt[0]>=0 and pt[1]<np.shape(mask)[1]-1 and pt[1]>=0) for pt in path]
        if False in not_in_img:
            not_in_img = np.where(not_in_img)
            path = np.delete(path,not_in_img,0)
        #next remove all values which are not in the mask
        not_in_mask = []
        for i in range (0,len(path)):
            pt = path[i]
            if mask[pt[0],pt[1]]:
                continue
            else:
                not_in_mask.append(i)
        return np.delete(path,not_in_mask,0)
    if np.any(np.isnan(np.array(poles))):
        raise ValueError('Poles must be numerical values, not np.NaN')
    #initializing parameters
    m = np.shape(skel)[0]
    n = np.shape(skel)[1]
    axis = m<n
    poles = list(poles)
    graph = sknw.build_sknw(skel) #creating the graph from the skeleton
    all_paths=shortest_path(graph) #shortest possible paths between the nodes in the skeleton
    nodes = graph.nodes()
    [start_pole,end_pole] = poles
    paths = []
    centerlines=[]
    lengths=[]
    
    #initializing suitable starting and ending positions for the skeleton
    if axis:
        start_nodes = list([i for i in nodes if nodes[i]['o'][1]<n//2])
        end_nodes= list([i for i in nodes if nodes[i]['o'][1]>n//2])
    else:
        start_nodes = list([i for i in nodes if nodes[i]['o'][0]<m//2])
        end_nodes= list([i for i in nodes if nodes[i]['o'][0]>m//2])
    
    # if there are some nodes close to the poles, check only those nodes
    true_starts = [i for i in start_nodes if np.linalg.norm(nodes[i]['o']-np.array([start_pole[1],start_pole[0]]))<sensitivity]
    true_ends = [i for i in end_nodes if np.linalg.norm(nodes[i]['o']-np.array([end_pole[1],end_pole[0]]))<sensitivity]
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
        #initializing centerline
        centerline_path = []
        #calling points from the graph
        edges = [(path[i],path[i+1]) for i in range (0,len(path)-1)]
        for (b,e) in edges:
            edge = graph[b][e]['pts']
            centerline_path = centerline_path + list(edge)
        centerline_path=order_by_NN(centerline_path,nodes[path[0]]['o'],nodes[path[-1]]['o'],max(m,n)/2)
        if len(centerline_path)==0:
            raise ValueError('Skeleton has been erased')
        if np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-end_pole) < np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-start_pole):
            centerline_path.reverse()
        centerline_path = crop_centerline(centerline_path)
        if len(centerline_path)<=5:
            raise ValueError('Skeleton has been erased')
        [xs,ys],u = find_splines(centerline_path)
        path = np.round(np.transpose(np.array([ys,xs]))).astype(np.uint32)
        path = in_mask(path,mask)
        length = arc_length([xs,ys])
        pruned_skel = pts_to_img(path,skel)
        return pruned_skel, length, [xs,ys],np.linspace(0,1,100)
    #convert paths (lists of nodes) to centerlines (lists of points)
    for path in paths:
        #initializing centerline
        centerline_path = []
        #calling points from the graph
        edges = [(path[i],path[i+1]) for i in range (0,len(path)-1)]
        for (b,e) in edges:
            edge = graph[b][e]['pts']
            centerline_path = centerline_path + list(edge)
        #convert path to binary image
        if len(centerline_path)==0:
            raise ValueError('Skeleton has been erased')
        #crop the centerline, if it has a false pole
        centerline_path=order_by_NN(centerline_path,nodes[path[0]]['o'],nodes[path[-1]]['o'],max(m,n)/2)
        if np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-end_pole) < np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-start_pole):
            centerline_path.reverse()
        centerline_path = crop_centerline(centerline_path)
        #find the length of each centerline
        length = len(centerline_path)
        # add to the list of centerlines and lengths
        centerlines.append(centerline_path)
        lengths.append(length)
    #choose the longest centerline
    max_index=lengths.index(max(lengths))
    centerline_path=centerlines[max_index]
    if len(centerline_path)<=5:
        raise ValueError('Skeleton has been erased')
    [xs,ys],u = find_splines(centerline_path)
    path = np.round(np.transpose(np.array([ys,xs]))).astype(np.uint32)
    path = in_mask(path,mask)
    pruned_skel = pts_to_img(path,skel)
    length = arc_length([xs,ys])
    return pruned_skel, length, [xs,ys], u

def radii(x,y):
    '''
    Finds the radius function of a 2d curve
    
    Parameters
    ---------
    x,y = coordinates of the curve
    
    Returns
    -------
    r = a 1D array containing the distance to the centroid of the curve for each x and y
    centroid = the centroid of the curve
    '''
    length = len(y)
    centroid = np.array([np.sum(x),np.sum(y)])/length
    vectors = np.transpose(np.array([x,y]))-centroid
    return np.array([np.linalg.norm(v) for v in vectors]),centroid

def explore_poles(outline,axis=True):
    '''
    Finds the poles (average distance of the farthest points from the centroid on a smooth closed curve) from x and y coords
    Parameters
    ----------
    outline = boolean array representing the outline of the cell
    axis = which axis the poles are found with respect to. True for the 0 (left-right) axis, False for the 1 (top-bottom) axis
    
    Returns
    --------
    poles 
    '''
    #find points. Want to start on the axis we wish to find them on
    outline_pts = bool_sort(outline,axis)
    outline_pts = order_by_NN(outline_pts,outline_pts[0],outline_pts[0])
    outline_pts = outline_pts + [outline_pts[0]]
    diff = np.diff(outline_pts,axis=0)==0
    repeated = np.where(np.array([pt[0] and pt[1] for pt in diff]))
    outline_pts = list(np.delete(outline_pts,repeated,axis=0))
    tck,U=splprep(np.transpose(outline_pts),per=1)
    [y,x] = splev(U,tck)
    r,centroid = radii(x,y)
    cx = centroid[0]
    cy = centroid[1]
    peaks = list(argrelmax(r)[0])
    if len(peaks)<2:
        peaks = peaks+[0,-1]
    if axis:
        right_x_pos=[x[i] for i in peaks if x[i]>cx]
        right_y_pos=[y[i] for i in peaks if x[i]>cx]
        right_rads=[r[i] for i in peaks if x[i]>cx]
        left_x_pos=[x[i] for i in peaks if x[i]<cx]
        left_y_pos=[y[i] for i in peaks if x[i]<cx]
        left_rads=[r[i] for i in peaks if x[i]<cx]
        
        left_pole = np.array([np.dot(left_x_pos,left_rads)/sum(left_rads),np.dot(left_y_pos,left_rads)/sum(left_rads)])
        right_pole = np.array([np.dot(right_x_pos,right_rads)/sum(right_rads),np.dot(right_y_pos,right_rads)/sum(right_rads)])
        return np.array([left_pole,right_pole]), centroid
    else:
        lower_x_pos=[x[i] for i in peaks if y[i]>cy]
        lower_y_pos=[y[i] for i in peaks if y[i]>cy]
        lower_rads=[r[i] for i in peaks if y[i]>cy]
        upper_x_pos=[x[i] for i in peaks if y[i]<cy]
        upper_y_pos=[y[i] for i in peaks if y[i]<cy]
        upper_rads=[r[i] for i in peaks if y[i]<cy]
        
        upper_pole = np.array([np.dot(upper_x_pos,upper_rads)/sum(upper_rads),np.dot(upper_y_pos,upper_rads)/sum(upper_rads)])
        lower_pole= np.array([np.dot(lower_x_pos,lower_rads)/sum(lower_rads),np.dot(lower_y_pos,lower_rads)/sum(lower_rads)])
        return list(np.transpose([upper_pole,lower_pole])), centroid

def arc_length(pts):
    '''
    Find the arclength of a curve given by a set of points
    Paramters
    --------
    pts = array-like coordinates [x1,x2,....]
    '''
    pts = np.array(np.transpose(pts))
    lengths = []
    for i in range (1,len(pts)):
        length = np.linalg.norm(pts[i] - pts[i-1])
        lengths.append(length)
    return np.sum(lengths)

def arc_length(pts):
    '''
    Find the arclength of a curve given by a set of points
    Paramters
    --------
    pts = array-like coordinates [x1,x2,....]
    '''
    pts = np.array(np.transpose(pts))
    lengths = []
    for i in range (1,len(pts)):
        length = np.linalg.norm(pts[i] - pts[i-1])
        lengths.append(length)
    return np.sum(lengths)