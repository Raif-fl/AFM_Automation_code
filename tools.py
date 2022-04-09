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
import skimage.morphology as sk_m
from fil_finder import FilFinder2D
from astropy import units as u
from radfil import radfil_class, styles
from shapely.geometry import Polygon

def atoi(text):
    '''
    A function which supports 
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    A function which allows me to sort input file names by timepoint/
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
        # append the bounding box to our list.
        boxes.append(box)
    return boxes

def get_overlap(outl, boxes):
    '''
    A function which iterates over all cell outlines in an image and detects how much of the cell's
    surface is overlapping another cell's. 
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

def peri_area(outl_list):
    '''
    An iterative function that calculates the perimeter and area for all cell outlines within a 
    list of images. 
    '''
    # Initialize lists to hold the perimeter and area information for all images.
    per_list = []
    area_list = []
    for outl in outl_list:
        # Initialize lists to hold the perimeter and area infromation within each individual image. 
        peris = []
        areas = []
        for o in outl:
            # Calculate the perimeter and area for each outline
            peris.append(cv2.arcLength(o, True))
            areas.append(cv2.contourArea(o))
        per_list.append(peris)
        area_list.append(areas)
    return per_list, area_list

def save_ind_masks(path, IDs_list, outl_new_list, time_list, img_list):
    '''
    A function which 
    '''
    for ID_set, outl, time, img in zip(IDs_list, outl_new_list, time_list, img_list):
        if os.path.exists(path + str(time)) == False:
            os.makedirs(path + str(time))
        if os.path.exists(path + str(time) + "/cells") == False:
            os.makedirs(path + str(time) + "/cells")
        if os.path.exists(path + str(time) + "/masks") == False:
            os.makedirs(path + str(time) + "/masks")
        if os.path.exists(path + str(time) + "/skeletons") == False:
            os.makedirs(path + str(time) + "/skeletons")
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
            
            # save the cell outline
            im_mask = Image.fromarray(out_mask)
            im_mask.save(path + str(time) + "/masks" + "/" + "mask_" + str(idx) + ".png")
            cv2.imwrite(path + str(time) + "/cells" + "/" + "cell_" + str(idx) + ".png", out_cell)
            
def radobj_maker(path, IDs_list, time_list):
    '''
    This iterative function loads up a set of images, masks and skeletons for individual cells which were created 
    from a set of parent images taken at different timepoints. For each of these individual cells, it then uses
    the radfil package to estimate the radial profile of the cell which is then saved in a list of radfil objects. 
    '''
    radobj_list = []
    for ID_set, time in zip(IDs_list, time_list):
        radobj_set = []
        print(time)
        for idx in (ID_set):
            # Load the image and convert to grayscale. 
            fil_image =imageio.imread(path + str(time) + "/cells/cell_" + str(idx) + ".png")
            fil_image = cv2.cvtColor(fil_image, cv2.COLOR_BGR2GRAY)
            
            # Load the mask and convert to grayscale
            fil_mask=imageio.imread(path + str(time) + "/masks/mask_" + str(idx) + ".png")
            fil_mask = cv2.cvtColor(fil_mask, cv2.COLOR_BGR2GRAY)
            fil_mask = np.array(fil_mask, dtype=bool)

            ###!!!### Replace with a call to Hasti's stuff
            spine = np.loadtxt(open(path + str(time) + "/skeletons/matrix_" + str(idx) + ".csv", "rb"), delimiter=",", skiprows=1)[1:-2].astype(int)
            dbl = int((len(spine)-2)/2)
            cut_spine = spine[1:-2]

            x, y = [i[0] for i in cut_spine], [i[1] for i in cut_spine]

            fil_spine = np.zeros(fil_mask.shape, dtype = bool)

            for i in range(len(cut_spine)):
                fil_spine[y[i], x[i]] = True
            
            # Use radfil to create a radial profil object and then append this object to a list.
            # We will use the data stored in this object to calculate height and radial profile along the
            # medial axis. 
            radobj=radfil_class.radfil(fil_image, mask=fil_mask, filspine=fil_spine, distance=200)
            radobj.build_profile(samp_int=1, shift = False)
            radobj_set.append(radobj)
        radobj_list.append(radobj_set)
    return(radobj_list)

def find_nearest(array, value):
    '''
    Finds the nearest value in an array and returns the index of that value.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def radobj_extractor(radobj_list, h_conv = 3.9215686274509802):
    '''
    This function iterates through a list of radfil objects and extracts two meaningful outputs for each cell.
    The first is a list that details the diameter of the cell along the medial axis, and the second is a list
    that details the pixel intensity of the original image along the medial axis. This second list will reflect
    whatever the original image was measuring (e.g. height, stiffness, brightness). 
    '''
    diam_list = []
    height_list = []
    for radobj_set in radobj_list:
        diam_set = []
        height_set = []
        for radobj in radobj_set:
            diam = []
            height = []
            for dist, prof in zip(radobj.dictionary_cuts['distance'],radobj.dictionary_cuts['profile']) :
                diam.append(np.abs(dist[0] - dist[-1]))
                ind = find_nearest(dist,0)
                height.append(prof[ind]*h_conv)
            diam_set.append(diam)
            height_set.append(height)
        diam_list.append(diam_set)
        height_list.append(height_set)
    return diam_list, height_list

def skeleton_length(path, IDs_list, time_list):
    '''
    This iterative function loads up a set of skeletons for individual cells which were created 
    from a set of parent images taken at different timepoints. For each of these skeletons, it uses
    the filfinder package to convert the skeleton to a filfinder skeleton. This fairly complicated
    process is done simply so that the length of the skeleton can be recorded. 
    '''
    length_list = []
    for ID_set, time in zip(IDs_list, time_list):
        length_set = []
        for idx in (ID_set):
             ###!!!### Replace with a call to Hasti's stuff
            spine = np.loadtxt(open(path + str(time) + "/skeletons/matrix_" + str(idx) + ".csv", "rb"), delimiter=",", skiprows=1)[1:-2].astype(int)
            dbl = int((len(spine)-2)/2)
            cut_spine = spine[1:-2]

            x, y = [i[0] for i in cut_spine], [i[1] for i in cut_spine]

            fil_mask=imageio.imread(path + str(time) + "/masks/mask_" + str(idx) + ".png")
            fil_spine = np.zeros(fil_mask.shape, dtype = 'uint8')

            for i in range(len(cut_spine)):
                fil_spine[y[i], x[i]] = 1
            fil_spine = cv2.cvtColor(fil_spine, cv2.COLOR_BGR2GRAY)
            
            fil = FilFinder2D(fil_spine, distance=250 * u.pix, mask=fil_spine)
            fil.preprocess_image(flatten_percent=85)
            fil.create_mask(border_masking=True, verbose=False,
            use_existing_mask=True)
            fil.medskel(verbose=False)
            fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
            if len(fil.lengths().value) > 0:
                length = fil.lengths().value[0]
            else:
                length = np.nan
        
            length_set.append(length)
        length_list.append(length_set)
    return(length_list)

def skeleton_length_old(path, IDs_list, time_list):
    '''
    This iterative function loads up a set of skeletons for individual cells which were created 
    from a set of parent images taken at different timepoints. For each of these skeletons, it uses
    the filfinder package to convert the skeleton to a filfinder skeleton. This fairly complicated
    process is done simply so that the length of the skeleton can be recorded. 
    '''
    length_list = []
    for ID_set, time in zip(IDs_list, time_list):
        length_set = []
        for idx in (ID_set):
             ###!!!### Replace with a call to Hasti's stuff
            fil_mask=imageio.imread(path + str(time) + "/masks/mask_" + str(idx) + ".png")
            skeleton = sk_m.skeletonize(fil_mask)
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)

            fil = FilFinder2D(skeleton, distance=250 * u.pix, mask=skeleton)
            fil.preprocess_image(flatten_percent=85)
            fil.create_mask(border_masking=True, verbose=False,
            use_existing_mask=True)
            fil.medskel(verbose=False)
            fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
            if len(fil.lengths().value) > 0:
                length = fil.lengths().value[0]
            else:
                length = np.nan
        
            length_set.append(length)
        length_list.append(length_set)
    return(length_list)

def get_max_ID(IDs_list):
    maxi = []
    for ID_set in IDs_list:
        maxi.append(max(ID_set))
    return max(maxi)

def get_metadata(exact_ID, IDs_list, per_list, area_list,
                 overl_list, centers_list, time_list,
                diam_list, height_list, length_list):
    data = []
    for ID_set, per_set, area_set, overl_set, centers_set, time, diam_set, height_set, length_set in zip(IDs_list,
                                    per_list, area_list, overl_list, centers_list, time_list,
                                    diam_list, height_list, length_list):
        if exact_ID in ID_set:
            for idx, per, area, overl, center, diam, height, length in zip(ID_set,
                                            per_set, area_set, overl_set, centers_set,
                                            diam_set, height_set, length_set):
                if idx == exact_ID:
                    data.append(dict(zip(["time", "perimeter", "area", "overl", "location",
                                     "diameter profile (pixels)", "Height (pixel intensity)", "length (pixels)"],
                                        [time, per, area, overl, center, diam, height, length])))
        else:
            data.append(dict(zip(["time", "perimeter", "area", "overl", "location",
                                     "diameter profile (pixels)", "Height (pixel intensity)", "length (pixels)"], 
                                 [time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])))
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset = "time", ignore_index = True)
    return(df)

# import the necessary packages
from collections import OrderedDict
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